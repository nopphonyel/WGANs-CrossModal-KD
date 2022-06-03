import os
import torch
import torch.nn as nn
import torchvision.transforms

from dataset.fMRI_HC import fMRI_HC_Dataset, DataLoader
from utils import save_model, mkdir
from utils.logger import LoggerGroup, Reporter

from libnn.transform import TanhRescale
from libnn.model.fe.alexextractor import AlexNetExtractor

# CONFIG section
ngpu = 1

dev = "cuda"
EPOCHS = 100
load_at_epoch = 0
LR = 1e-4
BS = 16
LOAD_GEN = False
LOAD_DIS = False

MODEL_EXPORT_NAME = "alexFID.pth"

# Define some path variable
__dirname__ = os.path.dirname(__file__)
MODEL_PATH = os.path.join(__dirname__, "export_content/saved_models/%s/" % fMRI_HC_Dataset.get_name())
# Also create a necessary directory to export stuff
mkdir(MODEL_PATH)

# Model declaration
net_alex_fid = AlexNetExtractor(output_class_num=6, feature_size=200, in_channel=1, pretrain=False).to(dev)
alex_optim = torch.optim.Adam(net_alex_fid.parameters(), lr=LR)

# Dataset declaration
img_tf = torch.nn.Sequential(
    torchvision.transforms.Resize((64, 64)),
    TanhRescale(min_in_val=0, max_in_val=255, margin_val=0.01)
)
ds = fMRI_HC_Dataset(p_id=1, v=1, train=True, img_tf=img_tf).to(dev)
ds_vali = fMRI_HC_Dataset(p_id=1, v=1, train=False, img_tf=img_tf).to(dev)
ld = DataLoader(ds, shuffle=True, batch_size=BS)
ld_vali = DataLoader(ds_vali, shuffle=True, batch_size=len(ds_vali))

# Loss function declaration
criterion = nn.CrossEntropyLoss()

# Logger declaration
alex_fid_lg = LoggerGroup("AlexFID Logger")

reporter = Reporter(alex_fid_lg, log_buffer_size=15)
reporter.set_experiment_name("AlexNet for FID")
reporter.append_summary_description(
    "This will and only train the AlexNetExtractor to recognize all of 5 letters from fMRI-HC dataset.")
reporter.append_summary_description("This model will be used to calculate FID for evaluating the performance of GANs")

if not reporter.review():
    quit()

try:
    for e in range(EPOCHS):
        # Training stuff
        net_alex_fid.train()
        for i, (_, img, lab) in enumerate(ld):
            lat, p_lab = net_alex_fid(img)
            loss_val = criterion(p_lab, lab)

            alex_optim.zero_grad()
            loss_val.backward()
            alex_optim.step()

            alex_fid_lg.collect_step('Loss', loss_val.item())
            reporter.report(epch=e, b_i=i, b_all=len(ld))

        # Eval stuff
        net_alex_fid.eval()
        for _, img_vali, lab_vali in ld_vali:
            lab, p_lab = net_alex_fid(img_vali)
            p_lab = torch.argmax(p_lab, dim=1)
            alex_acc = (torch.sum(p_lab == lab_vali) / p_lab.shape[0]) * 100.0

            alex_fid_lg.collect_step('Acc', alex_acc.item())

        alex_fid_lg.flush_step_all()
        if alex_fid_lg.get_value('last', 'Acc') == alex_fid_lg.get_value('max', 'Acc'):
            reporter.log("New high Acc has been detected: %.2f -> Exporting model: %s" % (
            alex_fid_lg.get_value('last', 'Acc'), MODEL_EXPORT_NAME))
            save_model(model=net_alex_fid, path=MODEL_PATH, filename=MODEL_EXPORT_NAME)

finally:
    reporter.stop()
    reporter.write_summary(os.path.dirname(__file__), f_name="alexFID.txt")
