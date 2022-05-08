from torch import nn


# custom weights initialization called on netG and netD
def weights_init(model):
    for m in model.modules():  # loop all layers in that model
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
