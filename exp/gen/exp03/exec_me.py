import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from torch.utils.data import DataLoader

import libnn.transform
import utils.path
from utils import *
from libnn.model import weights_init, wgans, wgans_kd
from libnn.model.wgans_kd import teachers
from libnn.model.wgans import gradient_penalty
from dataset.fMRI_HC import fMRI_HC_Dataset
from libnn.model.fe import *

import matplotlib.pyplot as plt
from utils.logger import LoggerGroup, Reporter

from exp.gen.gen_common_config import GenCommonConfig
from libnn.metrics.fid import FID, FIDOrig

# CONFIG section
ngpu = 1

dev = get_freer_gpu()
EPOCHS = 6000
load_at_epoch = 0
LR = 1e-4
BS = 16
LOAD_GEN = False
LOAD_DIS = False

z_dim = 100
lambda_gp = 10

d_epoch_steps = 5
g_epoch_steps = 1

preview_gen_num = 10
export_gen_img_every = 20

FID_CALC_FREQ = GenCommonConfig.FID_CALC_FREQ

# Define some path variable
__dirname__ = os.path.dirname(__file__)
MODEL_PATH = os.path.join(__dirname__, "export_content/saved_models/%s/" % fMRI_HC_Dataset.get_name())
IMAGE_PATH = os.path.join(__dirname__, "export_content/images/%s/" % fMRI_HC_Dataset.get_name())
WEIG_COL_PATH = utils.path.get_weights_collection_path()
# Also create a necessary directory to export stuff
mkdir(MODEL_PATH)
mkdir(IMAGE_PATH)

# Set random seeds
manualSeed = 794
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# ----Dataset declaration----
img_tf = nn.Sequential(
    torchvision.transforms.Resize((64, 64)),
    libnn.transform.TanhRescale(min_in_val=0, max_in_val=255, margin_val=0.01)
)

ds = fMRI_HC_Dataset(p_id=1, v=1, img_tf=img_tf).to(dev)
ds_val = fMRI_HC_Dataset(p_id=1, v=1, img_tf=img_tf, train=False).to(dev)

ld = DataLoader(ds, batch_size=BS, shuffle=True)
ld_val = DataLoader(ds_val, batch_size=len(ds_val), shuffle=True)

# ----Model declaration----
# - Classifier|Extractor section
non_img_extr = SimpleFCExtractor(in_features=948, num_layers=4, num_classes=6, latent_idx=2, latent_size=200)
img_extr = AlexNetExtractor(output_class_num=6, in_channel=1, feature_size=200, pretrain=False)

load_model(non_img_extr, path=MODEL_PATH, filename="non_img_extr.pth")
load_model(img_extr, path=WEIG_COL_PATH, filename="alexFID.pth")

non_img_extr.to(dev).eval()
img_extr.to(dev).eval()

# nimg_optim = torch.optim.Adam(non_img_extr.parameters(), lr=LR)
# img_optim = torch.optim.Adam(img_extr.parameters(), lr=LR)

# Metrics function declaration
fid = FID()
fid_orig = FIDOrig()

# Loss function declaration
criterion = nn.CrossEntropyLoss()


def j1_loss(l1, l2, f1, f2):
    """
    This loss implementation is following Dan Li et al works.
    According to the Dan Li et al. implementation:
        J1 (f1, f2) = (non-img latent, img_paired latent)
        J4 (f1, f2) = (non-img latent, img_unpaired latent)
    :param l1: A one hot encoded label
    :param l2: Another one hot encoded label
    :param f1: A latent (From paper, they use non-image) but I think both can be swap
    :param f2: Another latent
    :return:
    """
    s = torch.matmul(l2, l1.transpose(1, 0))
    delta = 0.5 * torch.matmul(torch.tanh(f1), torch.tanh(f2.transpose(1, 0)))
    losses = torch.mul(s, delta) - delta  # torch.log(torch.exp(delta)) why don't we just delta?
    loss = torch.mean(losses)
    return loss


# - WGANs section
num_classes = ds.get_num_classes()
netD = wgans.Discriminator(ngpu=1, num_classes=num_classes, latent_size=200, img_channel=1).to(dev)
netG = wgans_kd.Generator3Layers(ngpu=1, num_classes=num_classes, z_dim=z_dim, latent_size=200, img_channel=1).to(dev)
# netG = wgans.Generator(ngpu=1, num_classes=num_classes, z_dim=z_dim, latent_size=200, img_channel=1).to(dev)

if (dev == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
    netG = nn.DataParallel(netG, list(range(ngpu)))

# - Init model with some weight or resume from previous training
if load_at_epoch != 0:
    if LOAD_GEN:
        netG.load_state_dict(torch.load(MODEL_PATH + "%d_G.pth" % load_at_epoch))
    if LOAD_DIS:
        netD.load_state_dict(torch.load(MODEL_PATH + "%d_D.pth" % load_at_epoch))
else:
    netD.apply(weights_init)
    netG.apply(weights_init)

d_optim = torch.optim.Adam(netD.parameters(), lr=LR, betas=(0.0, 0.9))
g_optim = torch.optim.Adam(netG.parameters(), lr=LR, betas=(0.0, 0.9))

# ----Value tracker declaration----
wgan_logger = LoggerGroup("WGANs")
fid_logger = LoggerGroup("FID")

reporter = Reporter(wgan_logger, fid_logger, log_buffer_size=15)

reporter.set_experiment_name("Generator 3 layers")
reporter.append_summary_description(
    "Mainly experiment on Generator: my own 3 layers Generator with ridiculously large kernel size")
reporter.append_summary_description("\nTraining on: %s" % dev)
reporter.append_summary_description("\nfMRI extractor: 4 layers SimpleFC")
reporter.append_summary_description("\nImage extractor: AlexNet")
reporter.append_summary_description("\nfMRI size = %f MB" % model_size_mb(non_img_extr))
reporter.append_summary_description("\nGenerator size = %f MB" % model_size_mb(netG))

# Show the experiment details and description before run the training script
if not reporter.review():
    quit()

try:
    best = {
        "fid": None
    }
    for e in range(EPOCHS):
        # Train session
        for i, (fmri, img, label_idx) in enumerate(ld):
            curr_bs = fmri.shape[0]

            # Train netD
            for _ in range(d_epoch_steps):
                fy_p, ly_p = non_img_extr(fmri)
                ly_p_idx = torch.argmax(ly_p, dim=1)

                zz = torch.randn(curr_bs, z_dim, 1, 1, device=dev)
                ld_real = netD(img, ly_p_idx, fy_p).view(-1)
                p_stem, l_01, l_03, l_05, fake_img = netG(zz, ly_p_idx, fy_p)
                ld_fake = netD(fake_img, ly_p_idx, fy_p).view(-1)

                gp = gradient_penalty(netD, fy_p, ly_p_idx, img, fake_img, dev)
                d_loss = -(torch.mean(ld_real) - torch.mean(ld_fake)) + lambda_gp * gp
                wgan_logger.collect_sub_step('d_loss', d_loss.item())

                d_optim.zero_grad()
                d_loss.backward(retain_graph=True)
                d_optim.step()

            # Train netG
            for _ in range(g_epoch_steps):
                fy_p, ly_p = non_img_extr(fmri)
                ly_p_idx = torch.argmax(ly_p, dim=1)

                zz = torch.randn(curr_bs, z_dim, 1, 1, device=dev)
                p_stem, l_01, l_03, l_05, fake_img = netG(zz, ly_p_idx, fy_p)
                ld_fake = netD(fake_img, ly_p_idx, fy_p).view(-1)

                g_loss = -torch.mean(ld_fake)

                # Test back prop again... in hope that discriminator also improve the non-img
                g_optim.zero_grad()
                # nimg_optim.zero_grad()
                g_loss.backward(retain_graph=True)
                g_optim.step()
                # nimg_optim.step()

                wgan_logger.collect_sub_step('g_loss', g_loss.item())
            wgan_logger.flush_sub_step_all()

            # Validate stuff
            for fmri_val, img_val, label_idx_val in ld_val:
                non_img_extr.eval()
                img_extr.eval()

                fy_p, ly_p = non_img_extr(fmri_val)
                _, lx_p = img_extr(img_val)
                ly_p_idx = torch.argmax(ly_p, dim=1)
                lx_p_idx = torch.argmax(lx_p, dim=1)

                # Only export image when end of 'export_gen_img_every'th epoch
                if i == 0 and (e + 1) % export_gen_img_every == 0:
                    zz = torch.randn(preview_gen_num, z_dim, 1, 1, device=dev)
                    fy_p_exp = fy_p[0:preview_gen_num, :]
                    ly_p_idx_exp = ly_p_idx[0:preview_gen_num]
                    p_stem, l_01, l_03, l_05, fake_img = netG(zz, ly_p_idx_exp, fy_p_exp)

                    real_img = make_grid(img_val[0:preview_gen_num, :, :, :], nrow=preview_gen_num, normalize=True)
                    fake_img = make_grid(fake_img, nrow=preview_gen_num, normalize=True)
                    img_grid = torch.cat((real_img, fake_img), 1)
                    save_image(img_grid, IMAGE_PATH + "epoch_{}.png".format(e), normalize=False)

                if i == 0 and (e + 1) % FID_CALC_FREQ == 0:
                    zz = torch.randn(fmri_val.shape[0], z_dim, 1, 1, device=dev)
                    p_stem, l_01, l_03, l_05, fake_img = netG(zz, ly_p_idx, fy_p)

                    gen_lat, _ = img_extr(fake_img)
                    real_lat, _ = img_extr(img_val)

                    fid_score = fid(real_lat, gen_lat)
                    fid_orig_score = fid_orig(real_batch=img_val, gen_batch=fake_img)
                    fid_logger.collect_step('AlexFID', fid_score)
                    fid_logger.collect_step('FIDIncep', fid_orig_score)

                    # Decided to save model or not
                    if best['FIDIncep'] is None or best['FIDIncep'] > fid_score:
                        best['FIDIncep'] = fid_score
                        reporter.log("Best FID detected: %.4f -> Exporting Generator" % fid_score)
                        save_model(netG, MODEL_PATH, "netG_3layers.pth")

            reporter.report(epch=e + 1, b_i=i, b_all=len(ld) - 1)

        wgan_logger.flush_step_all()
        fid_logger.flush_step_all()

    wgan_logger.report()
    fid_logger.report()

finally:
    reporter.stop()
    reporter.write_summary(os.path.dirname(__file__), f_name="summary_gen_3layers.txt")
