import torch
from torch import nn


class TinyGenerator(nn.Module):
    """
    This model will reduce teacher model (GeneratorKD) which has 5 layers -> 3 layers
    """
    def __init__(self, gen_dim=64):
        super(TinyGenerator, self).__init__()

    def forward(self, x):
        return x


from libnn.model.wgans_kd.teachers import GeneratorKD

gen_t = GeneratorKD(num_classes=6)