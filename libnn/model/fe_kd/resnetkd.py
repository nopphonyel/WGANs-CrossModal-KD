from libnn.model.fe.resnetextractor import *


class ResNet18KD(ResNet18Extractor):
    def __init__(self, in_features, out_features, num_classes, dropout_p=0.0):
        super(ResNet18KD, self).__init__(in_features, out_features, num_classes, dropout_p=0.0)

    def forward(self, x):
        shaped = self.shaper(x)
        l_0 = self.l0(shaped)
        l_1 = self.l1(l_0)
        l_2 = self.l2(l_1)
        l_3 = self.l3(l_2)
        l_4 = self.l4(l_3)
        gap = self.gap(l_4)
        gap_fl = torch.flatten(gap, start_dim=1)
        latent = self.fc(gap_fl)
        class_p = self.final_fc(latent)
        return shaped, l_0, l_1, l_2, l_3, l_4, gap, gap_fl, latent, class_p



