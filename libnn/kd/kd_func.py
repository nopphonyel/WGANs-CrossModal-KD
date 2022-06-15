import torch
import torch.nn as nn
import torch.nn.functional as F
from libnn.kd.kd_utils import Embed, ContrastLoss, ContrastMemory


class SoftTarget(nn.Module):
    """
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    """

    def __init__(self, temperature):
        """
        :param temperature: Temperature term which used in soft target
        """
        super(SoftTarget, self).__init__()
        self.T = temperature

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s / self.T, dim=1),
                        F.softmax(out_t / self.T, dim=1),
                        reduction='batchmean') * self.T * self.T
        return loss


class Logits(nn.Module):
    '''
    Do Deep Nets Really Need to be Deep?
    http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf
    '''

    def __init__(self):
        super(Logits, self).__init__()

    def forward(self, out_s, out_t):
        loss = F.mse_loss(out_s, out_t)
        return loss


class AT(nn.Module):
    '''
    Paying More Attention to Attention: Improving the Performance of Convolutional
    Neural Netkworks wia Attention Transfer
    https://arxiv.org/pdf/1612.03928.pdf
    '''

    def __init__(self, p):
        super(AT, self).__init__()
        self.p = p

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))
        return loss

    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        am = torch.div(am, norm + eps)
        return am


class AFD(nn.Module):
    '''
    In the original paper, AFD is one of components of AFDS.
    AFDS: Attention Feature Distillation and Selection
    AFD:  Attention Feature Distillation
    AFS:  Attention Feature Selection

    We (not me) find the original implementation of attention is unstable, thus we replace it with a SE block.

    Pay Attention to Features, Transfer Learn Faster CNNs
    https://openreview.net/pdf?id=ryxyCeHtPB
    '''

    def __init__(self, in_channels, att_f):
        super(AFD, self).__init__()
        mid_channels = int(in_channels * att_f)

        self.attention = nn.Sequential(*[
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, 1, 0, bias=True)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, fm_s, fm_t, eps=1e-6):
        fm_t_pooled = F.adaptive_avg_pool2d(fm_t, 1)
        rho = self.attention(fm_t_pooled)
        # rho = F.softmax(rho.squeeze(), dim=-1)
        rho = torch.sigmoid(rho.squeeze())
        rho = rho / torch.sum(rho, dim=1, keepdim=True)  # Looks like an average here

        fm_s_norm = torch.norm(fm_s, dim=[2, 3], keepdim=True)
        fm_s = torch.div(fm_s, fm_s_norm + eps)
        fm_t_norm = torch.norm(fm_t, dim=[2, 3], keepdim=True)
        fm_t = torch.div(fm_t, fm_t_norm + eps)

        loss = rho * torch.pow(fm_s - fm_t, 2).mean(dim=(2, 3))
        loss = loss.sum(1).mean(0)

        return loss


'''

'''


class CRD(nn.Module):
    '''
    Both CRD and it's component are modified from
    -> https://github.com/HobbitLong/RepDistiller/tree/master/crd

    Contrastive Representation Distillation
    -> https://openreview.net/pdf?id=SkgpBJrtvS

    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side

    Args:
        s_dim: the dimension of student's feature
        t_dim: the dimension of teacher's feature
        feat_dim: the dimension of the projection space
        nce_n: number of negatives paired with each positive
        nce_t: the temperature
        nce_mom: the momentum for updating the memory buffer
        n_data: the number of samples in the training set, which is the M in Eq.(19)
    '''

    def __init__(self, s_dim, t_dim, feat_dim, nce_n, nce_t, nce_mom, n_data):
        super(CRD, self).__init__()
        self.embed_s = Embed(s_dim, feat_dim)
        self.embed_t = Embed(t_dim, feat_dim)
        self.contrast = ContrastMemory(feat_dim, n_data, nce_n, nce_t, nce_mom)
        self.criterion_s = ContrastLoss(n_data)
        self.criterion_t = ContrastLoss(n_data)

    def forward(self, feat_s, feat_t, idx, sample_idx):
        feat_s = self.embed_s(feat_s)
        feat_t = self.embed_t(feat_t)
        out_s, out_t = self.contrast(feat_s, feat_t, idx, sample_idx)
        loss_s = self.criterion_s(out_s)
        loss_t = self.criterion_t(out_t)
        loss = loss_s + loss_t

        return loss
