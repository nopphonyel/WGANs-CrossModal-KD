import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from torch.utils.data import DataLoader

import libnn.transform
from utils import *
from libnn.model import weights_init, wgans
from libnn.model.wgans import gradient_penalty
from dataset.fMRI_HC import fMRI_HC_Dataset
from libnn.model.fe import *

import matplotlib.pyplot as plt

from utils.logger import LoggerGroup, Reporter

# CONFIG section
ngpu = 1

dev = "cuda"
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

preview_gen_num = 20
export_gen_img_every = 20

SUMMARY_F_NAME = "summary_fc4l.txt"

# Define some path variable
__dirname__ = os.path.dirname(__file__)
MODEL_PATH = os.path.join(__dirname__, "export_content/saved_models/%s/" % fMRI_HC_Dataset.get_name())
IMAGE_PATH = os.path.join(__dirname__, "export_content/images/%s/" % fMRI_HC_Dataset.get_name())
# Also create a necessary directory to export stuff
mkdir(MODEL_PATH)

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
non_img_extr = SimpleFCExtractor(in_features=948, num_layers=4, num_classes=6, latent_idx=2, latent_size=200).to(dev)
img_extr = AlexNetExtractor(output_class_num=6, in_channel=1, feature_size=200, pretrain=False).to(dev)

nimg_optim = torch.optim.Adam(non_img_extr.parameters(), lr=LR)
img_optim = torch.optim.Adam(img_extr.parameters(), lr=LR)

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

# ----Value tracker declaration----
loss_logger = LoggerGroup("Loss")
acc_logger = LoggerGroup("Accuracy")

reporter = Reporter(loss_logger, acc_logger, log_buffer_size=10)

reporter.set_experiment_name("CrossModal Classifier SimpleFC (4 layers)")
reporter.append_summary_description("Using fMRI-FE that based on SimpleFC (4 layers, latent@layer[2]).")
reporter.append_summary_description("\nfMRI size = %f MB" % model_size_mb(non_img_extr))

# Show the experiment details and description before run the training script
if not reporter.review():
    quit()

try:
    max_acc = {
        'j2': 0.0,
        'j3': 0.0
    }
    for e in range(EPOCHS):
        # Train session
        for i, (fmri, img, label_idx) in enumerate(ld):
            curr_bs = fmri.shape[0]

            l_p = F.one_hot(label_idx, num_classes=6).float()
            fy_p, ly_p = non_img_extr(fmri)
            fx_p, lx_p = img_extr(img)

            j1 = j1_loss(l_p, l_p, fy_p, fx_p)
            j2 = criterion(ly_p, label_idx)
            j3 = criterion(lx_p, label_idx)

            loss = j1 + j2 + j3

            nimg_optim.zero_grad()
            img_optim.zero_grad()
            loss.backward()
            nimg_optim.step()
            img_optim.step()

            # Report
            loss_logger.collect_step('total', loss.item())
            loss_logger.collect_step('j1', j1.item())
            loss_logger.collect_step('j2', j2.item())
            loss_logger.collect_step('j3', j3.item())

            # Validate stuff
            for fmri_val, img_val, label_idx_val in ld_val:
                non_img_extr.eval()
                img_extr.eval()

                fy_p, ly_p = non_img_extr(fmri_val)
                _, lx_p = img_extr(img_val)
                ly_p_idx = torch.argmax(ly_p, dim=1)
                lx_p_idx = torch.argmax(lx_p, dim=1)
                j2_acc = torch.sum(ly_p_idx == label_idx_val) / label_idx_val.shape[0]
                j3_acc = torch.sum(lx_p_idx == label_idx_val) / label_idx_val.shape[0]

                j2_acc = j2_acc.item() * 100
                j3_acc = j3_acc.item() * 100

                if max_acc['j2'] < j2_acc:
                    max_acc['j2'] = j2_acc
                    reporter.log("New high j2_acc detected! = %.2f Exporting model: non_img_extr" % j2_acc)
                    save_model(non_img_extr, path=MODEL_PATH, filename="non_img_extr.pth")

                if max_acc['j3'] < j3_acc:
                    max_acc['j3'] = j3_acc
                    reporter.log("New high j3_acc detected! = %.2f Exporting model: img_extr" % j3_acc)
                    save_model(img_extr, path=MODEL_PATH, filename="img_extr.pth")

                acc_logger.collect_step('j2_acc', j2_acc)
                acc_logger.collect_step('j3_acc', j3_acc)

            non_img_extr.train()
            img_extr.train()

            reporter.report(epch=e + 1, b_i=i, b_all=len(ld) - 1)

        loss_logger.flush_step_all()
        acc_logger.flush_step_all()

    loss_logger.report()
    acc_logger.report()

finally:
    reporter.stop()
    reporter.write_summary(os.path.dirname(__file__), f_name=SUMMARY_F_NAME)
