import torch
from torch import nn
from libnn import DeptSepConvTranspose2d


class Generator3Layers(nn.Module):
    """
    This model will reduce teacher model (GeneratorKD) which has 5 layers -> 3 layers
    """

    def __init__(self, ngpu, num_classes, embed_size=100, z_dim=100, latent_size=200, img_channel=3, gen_dim=64):
        super(Generator3Layers, self).__init__()
        self.ngpu = ngpu
        self.img_channel = img_channel

        self.b_01 = self._block(z_dim + embed_size + latent_size, gen_dim * 16, 4, 1, 0)
        self.b_03 = self._block(gen_dim * 16, gen_dim * 4, 8, 4, 2)  # batch x 256 x 16 x 16

        # did not use block because the last layer won't use batch norm or relu
        self.b_05 = nn.ConvTranspose2d(gen_dim * 4, self.img_channel, kernel_size=8, stride=4, padding=2)
        self.tanh = nn.Tanh()
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            DeptSepConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,  # batch norm does not require bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)  # in_place = True
        )

    def forward(self, zz, labels, semantic_latent):
        semantic_latent = semantic_latent.unsqueeze(2).unsqueeze(3)  # batch, feature_size, 1, 1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)  # batch, embed_size, 1, 1
        p_stem = torch.cat([zz, embedding, semantic_latent], dim=1)
        l_01 = self.b_01(p_stem)
        l_03 = self.b_03(l_01)
        l_05 = self.b_05(l_03)
        out = self.tanh(l_05)
        return p_stem, l_01, l_03, l_05, out


class GeneratorDeptSep(nn.Module):
    """
    Replace the traditional ConvTranspose2d to DeptSepConvTranspose2d
    to reduce the numbers of parameters
    """

    def __init__(self, ngpu, num_classes, embed_size=100, z_dim=100, latent_size=200, img_channel=3, gen_dim=64):
        """
        :param ngpu:
        :param num_classes: number of image class (use for do label embedding)
        :param embed_size: size of image label embedded vector
        :param z_dim: input noise vector
        :param latent_size: size of feature vector
        :param img_channel: an output image channel (default=3)
        :param gen_dim: number of dimension factor (default=64)
        """
        super(GeneratorDeptSep, self).__init__()
        self.ngpu = ngpu
        self.img_channel = img_channel

        self.b_01 = self._block(z_dim + embed_size + latent_size, gen_dim * 16, 4, 1, 0)
        self.b_02 = self._block(gen_dim * 16, gen_dim * 8, 4, 2, 1)  # batch x 512 x 8 x 8
        self.b_03 = self._block(gen_dim * 8, gen_dim * 4, 4, 2, 1)  # batch x 256 x 16 x 16
        self.b_04 = self._block(gen_dim * 4, gen_dim * 2, 4, 2, 1)  # batch x 128 x 32 x 32

        # did not use block because the last layer won't use batch norm or relu
        self.b_05 = nn.ConvTranspose2d(gen_dim * 2, self.img_channel, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            DeptSepConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,  # batch norm does not require bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)  # in_place = True
        )

    def forward(self, zz, labels, semantic_latent):
        semantic_latent = semantic_latent.unsqueeze(2).unsqueeze(3)  # batch, feature_size, 1, 1
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)  # batch, embed_size, 1, 1
        p_stem = torch.cat([zz, embedding, semantic_latent], dim=1)
        l_01 = self.b_01(p_stem)
        l_02 = self.b_02(l_01)
        l_03 = self.b_03(l_02)
        l_04 = self.b_04(l_03)
        l_05 = self.b_05(l_04)
        out = self.tanh(l_05)
        return p_stem, l_01, l_02, l_03, l_04, l_05, out


if __name__ == "__main__":
    from libnn.model.wgans_kd.teachers import GeneratorKD
    from utils import model_size_mb

    lat = torch.rand([1, 200])
    noise = torch.rand([1, 100, 1, 1])
    lab = torch.randint(low=0, high=5, size=[1])

    gen_t = GeneratorKD(ngpu=1, num_classes=6)
    gen_s_ds = GeneratorDeptSep(ngpu=1, num_classes=6)
    gen_s_3l = Generator3Layers(ngpu=1, num_classes=6)

    _, l1, _, l3, _, l5, out = gen_t(noise, lab, lat)
    _, l1s, _, l3s, _, l5s, outs = gen_s_ds(noise, lab, lat)

    print("_", l3.shape, ": s", l3s.shape)
    print("_", l5.shape, ": s", l5s.shape)
    print("Model size")
    print(
        "_ :", model_size_mb(gen_t),
        "\ns_ds :", model_size_mb(gen_s_ds),
        "\ns_3l :", model_size_mb(gen_s_3l)
    )
    pass
