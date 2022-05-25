import torch.nn as nn


class KDLoss(nn.Module):
    def __init__(self):
        super(KDLoss, self).__init__()

    def forward(self, fp_t: list, fp_s: list):
        """
        param feature_pack: list of output features
        """
