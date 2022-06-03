from torch import nn


class TanhRescale(nn.Module):
    """
    Rescale the value tensor to between -1 and 1
    """

    def __init__(self, min_in_val, max_in_val, margin_val=0):
        """
        :param min_in_val: Min value in the tensor that want to normalize
        :param max_in_val: Max value in the tensor
        :param margin_val: Sometimes, Add some margin to output. This might help Tanh function to reach those value.
        """
        super(TanhRescale, self).__init__()
        self.mn = min_in_val - margin_val
        self.db2 = ((max_in_val + margin_val) - min_in_val) / 2.0

    def forward(self, x):
        return ((x - self.mn) / self.db2) - 1.0


class GreyScaleToRGB(nn.Module):
    """
    Convert any grey scale image tensor to rgb image tensor
    """

    def __init__(self):
        super(GreyScaleToRGB, self).__init__()

    def forward(self, x):
        return x.repeat(3, 1, 1)
