import os
import libnn
import torch
import torchvision
import torch.nn as nn

from utils import load_model, save_model, mkdir, path, get_freer_gpu
from utils.logger import LoggerGroup, Reporter

from dataset.fMRI_HC import fMRI_HC_Dataset, DataLoader

from libnn.kd import kd_func
from libnn.metrics.fid import FID, FIDOrig
from libnn.model.fe.alexextractor import AlexNetExtractor
from libnn.model.fe.simplcfcextractor import SimpleFCExtractor
from libnn.model.wgans_kd.teachers import GeneratorKD, Discriminator
from libnn.model.wgans_kd.students import GeneratorDeptSep, Generator3Layers

from exp.kd.kd_common_config import KDCommonConfig

# Define some config
nGPU = KDCommonConfig.nGPU
DEV = get_freer_gpu()

# Training config
EPOCHS = 6000
BS = KDCommonConfig.BS
LR = KDCommonConfig.LR
BETAS = KDCommonConfig.BETAS_G

# GANs configuration
# > Generator
NUM_CLASS = KDCommonConfig.NUM_CLASS
Z_DIM = KDCommonConfig.Z_DIM
LATENT_SIZE = KDCommonConfig.LATENT_SIZE
IMG_CHAN = KDCommonConfig.IMG_CHAN
# > Discriminator
IMG_SIZE = 64

# KD Config
AT_P = 2.0
LAMBDA_KD = 1.0

# Define some path variable
__dirname__ = os.path.dirname(__file__)
MODEL_PATH = os.path.join(__dirname__, "export_content/saved_models/%s/" % fMRI_HC_Dataset.get_name())
IMAGE_PATH = os.path.join(__dirname__, "export_content/images/%s/" % fMRI_HC_Dataset.get_name())
WEIG_COL_PATH = path.get_weights_collection_path()
WEIG_COL_KD_PATH = path.get_weights_collection_path(sub_dir="kd")
# Also create a necessary directory to export stuff
mkdir(MODEL_PATH)
mkdir(IMAGE_PATH)

# Define model
# > Generators
netG_t = GeneratorKD(
    ngpu=nGPU,
    num_classes=NUM_CLASS,
    z_dim=Z_DIM,
    latent_size=LATENT_SIZE,
    img_channel=IMG_CHAN).to(DEV)
load_model(model=netG_t, path=WEIG_COL_KD_PATH, filename='netG_t.pth')
# netG_s = GeneratorDeptSep(
#     ngpu=nGPU,
#     num_classes=NUM_CLASS,
#     z_dim=Z_DIM,
#     latent_size=LATENT_SIZE,
#     img_channel=IMG_CHAN).to(DEV)
netG_s = Generator3Layers(
    ngpu=nGPU,
    num_classes=NUM_CLASS,
    z_dim=Z_DIM,
    latent_size=LATENT_SIZE,
    img_channel=IMG_CHAN).to(DEV)
# > Discriminators
netD_t = Discriminator(
    ngpu=nGPU,
    num_classes=NUM_CLASS,
    img_size=IMG_SIZE,
    latent_size=LATENT_SIZE,
    img_channel=IMG_CHAN
).to(DEV)
load_model(netD_t, WEIG_COL_KD_PATH, 'netD_t.pth')

non_img_extr = SimpleFCExtractor(in_features=948, num_layers=4, num_classes=6, latent_idx=2, latent_size=200)
load_model(non_img_extr, WEIG_COL_KD_PATH, 'non_img_extr_t.pth')
fid_extr = AlexNetExtractor(output_class_num=6, in_channel=1, feature_size=200, pretrain=False)
load_model(fid_extr, WEIG_COL_PATH, 'alexFID.pth')

non_img_extr.to(DEV).eval()
fid_extr.to(DEV).eval()

# Define Optimizer for each model
g_s_optim = torch.optim.Adam(netG_s.parameters(), lr=LR, betas=BETAS)

# Define dataset
img_tf = nn.Sequential(
    torchvision.transforms.Resize((64, 64)),
    libnn.transform.TanhRescale(min_in_val=0, max_in_val=255, margin_val=0.01)
)

ds = fMRI_HC_Dataset(p_id=1, v=1, train=True, img_tf=img_tf).to(DEV)
ds_val = fMRI_HC_Dataset(p_id=1, v=1, train=False, img_tf=img_tf).to(DEV)
ld = DataLoader(ds, batch_size=BS, shuffle=True)
ld_val = DataLoader(ds_val, batch_size=len(ds_val), shuffle=True)

# Define some loss func
kd_loss_logits = kd_func.Logits()
kd_loss_at = kd_func.AT(p=AT_P)

# Define eval metrics
fid = FID()
fid_orig = FIDOrig().to(DEV)

d_t_loss = LoggerGroup(title="Gen loss by Discrim")
total_kd_lg = LoggerGroup(title="Total KD Losses")
kd_loss_lg = LoggerGroup(title="KD Losses")
fid_lg = LoggerGroup(title="FID AlexNet")
fid_orig_lg = LoggerGroup(title="FID Original")
reporter = Reporter(kd_loss_lg, d_t_loss, total_kd_lg, fid_lg, fid_orig_lg, log_buffer_size=15)
reporter.set_experiment_name("KD on 3LayersGenerator - 1stAT+Discrim Loss")
reporter.set_summary_description("Using the first feature layer for calculating AT_KD loss")
reporter.append_summary_description("\n\t > Train on dev: %s" % DEV)

if not reporter.review():
    quit()

try:
    for e in range(EPOCHS):

        # Training phase
        for i, (fmri, img, lab) in enumerate(ld):
            # Set model(s) to training mode
            netG_s.train()

            curr_bs = fmri.shape[0]

            # Feed to both generators
            fy_p, ly_p = non_img_extr(fmri)
            ly_p_idx = torch.argmax(ly_p, dim=1)

            zz = torch.randn(curr_bs, Z_DIM, 1, 1, device=DEV)
            p_stem_t, l_01_t, _, l_03_t, _, l_05_t, fake_img_t = netG_t(zz, ly_p_idx, fy_p)
            p_stem_s, l_01_s, l_03_s, l_05_s, fake_img_s = netG_s(zz, ly_p_idx, fy_p)
            ld_fake = netD_t(fake_img_s, ly_p_idx, fy_p).view(-1)

            g_loss = -torch.mean(ld_fake)
            # kd_loss_logits_val = kd_loss_logits(fake_img_t, fake_img_s)
            kd_loss_at_val = [
                kd_loss_at(l_01_s, l_01_t),
                # kd_loss_at(l_03_t, l_03_s),
                # kd_loss_at(l_05_t, l_05_s)
            ]

            kd_loss_total_val = g_loss + (sum(kd_loss_at_val) / len(kd_loss_at_val)) * LAMBDA_KD

            # Test back prop again... in hope that discriminator also improve the non-img
            g_s_optim.zero_grad()
            # nimg_optim.zero_grad()
            kd_loss_total_val.backward(retain_graph=True)
            g_s_optim.step()
            # nimg_optim.step()
            # kd_loss_lg.collect_step('logits', kd_loss_logits_val.item())
            kd_loss_lg.collect_step('at_l1', kd_loss_at_val[0].item())
            # kd_loss_lg.collect_step('at_l2', kd_loss_at_val[1].item())
            # kd_loss_lg.collect_step('at_l3', kd_loss_at_val[2].item())
            d_t_loss.collect_step('->', g_loss.item())
            total_kd_lg.collect_step('->', kd_loss_total_val.item())

            reporter.report(epch=e, b_i=i, b_all=len(ld))

        # Validation phase
        for fmri_val, img_val, lab_val in ld_val:
            curr_bs = fmri_val.shape[0]

            # Feed to both generators
            fy_p, ly_p = non_img_extr(fmri_val)
            ly_p_idx = torch.argmax(ly_p, dim=1)

            zz = torch.randn(curr_bs, Z_DIM, 1, 1, device=DEV)
            _, _, _, _, _, _, fake_img_t = netG_t(zz, ly_p_idx, fy_p)
            _, _, _, _, fake_img_s = netG_s(zz, ly_p_idx, fy_p)

            real_feature, _ = fid_extr(img_val)
            teacher_feature, _ = fid_extr(fake_img_t)
            student_feature, _ = fid_extr(fake_img_s)

            fid_value = fid(teacher_feature, student_feature)
            fid_orig_value = fid_orig(real_batch=fake_img_t, gen_batch=fake_img_s)
            fid_lg.collect_step('T<->S', fid_value.item())
            fid_orig_lg.collect_step('T<->S', fid_orig_value.item())

            fid_value = fid(real_feature, student_feature)
            fid_orig_value = fid_orig(real_batch=img_val, gen_batch=fake_img_s)
            fid_lg.collect_step('S<->R', fid_value.item())
            fid_orig_lg.collect_step('S<->R', fid_orig_value.item())

            fid_value = fid(real_feature, teacher_feature)
            fid_orig_value = fid_orig(real_batch=img_val, gen_batch=fake_img_t)
            fid_lg.collect_step('T<->R', fid_value.item())
            fid_orig_lg.collect_step('T<->R', fid_orig_value.item())

            if fid_lg.get_value('last', 'S<->R') <= fid_lg.get_value('min', 'S<->R'):
                reporter.log("Best generator has been found... Exporting model netG_s.pth")
                reporter.log("  > Info [EPCH=%d] [FID(S<->R)=%.4f]" % (e, fid_lg.get_value('last', 'S<->R')))
                save_model(netG_s, path=MODEL_PATH, filename="netG_s.pth")

        d_t_loss.flush_step_all()
        total_kd_lg.flush_step_all()
        fid_lg.flush_step_all()
        fid_orig_lg.flush_step_all()
        kd_loss_lg.flush_step_all()


finally:
    reporter.stop()
    reporter.write_summary(os.path.dirname(__file__), f_name="kd_at_summary.txt")
