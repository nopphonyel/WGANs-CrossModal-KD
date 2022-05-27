import torch
from libnn.model.wgans import Generator, Discriminator


class GeneratorKD(Generator):
    def __init__(self, ngpu, num_classes, embed_size=100, z_dim=100, latent_size=200, img_channel=3, gen_dim=64):
        super(GeneratorKD, self).__init__(self,
                                          ngpu,
                                          num_classes,
                                          embed_size=embed_size,
                                          z_dim=z_dim,
                                          latent_size=latent_size,
                                          img_channel=img_channel,
                                          gen_dim=gen_dim
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
