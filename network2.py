import torch
import torch.nn as nn


class BlockDown(nn.Module):
    def __init__(self, in_ch, out_ch, ks=4, s=2, p=1, relu=True, norm=True, bias=True):
        super(BlockDown, self).__init__()

        layers = []
        if relu:
            layers += [nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks, stride=s, padding=p, bias=bias)]
        if norm:
            layers += [nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        return x


class BlockCenter(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, s=1, p=1, relu=True, norm=True, bias=True):  # ks=1, s=1, p=0
        super(BlockCenter, self).__init__()

        layers = []
        if relu:
            layers += [nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks, stride=s, padding=p, bias=bias)]
        if norm:
            layers += [nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*layers)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        x = self.conv(x)
        return x
    

class BlockUp(nn.Module):
    def __init__(self, in_ch, out_ch, ks=4, s=2, p=1, norm=True, drop=True, bias=True):
        super(BlockUp, self).__init__()

        layers = [
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks, stride=s, padding=p, bias=bias)
        ]
        if norm:
            layers += [nn.BatchNorm2d(out_ch)]
        if drop:
            layers += [nn.Dropout(0.5)]

        self.conv = nn.Sequential(*layers)

    # def forward(self, x1, x2=None):
    #     x = x1 if x2 is None else torch.cat([x1, x2], 1)
    #     x = self.conv(x)
    #     return x

    def forward(self, x1, *x2):
        x = x1
        for data in x2:
            x = torch.cat([x, data], 1)
        x = self.conv(x)
        return x


class HideNet1(nn.Module):
    # initializers
    def __init__(self, in_ch=6, out_ch=3, d=64):
        super(HideNet1, self).__init__()
        # Unet encoder
        self.conv1 = BlockDown(in_ch, d, 4, 2, 1, relu=False, norm=False)
        self.conv2 = BlockDown(d, d * 2, 4, 2, 1)
        self.conv3 = BlockDown(d * 2, d * 4, 4, 2, 1)
        self.conv4 = BlockDown(d * 4, d * 8, 4, 2, 1)
        self.conv5 = BlockDown(d * 8, d * 8, 4, 2, 1)
        self.conv6 = BlockDown(d * 8, d * 8, 4, 2, 1)
        self.conv7 = BlockDown(d * 8, d * 8, 4, 2, 1, norm=False)

        # Unet decoder
        self.deconv1 = BlockUp(d * 8, d * 8, 4, 2, 1, drop=False)
        self.deconv2 = BlockUp(d * 8 * 2, d * 8, 4, 2, 1, drop=False)
        self.deconv3 = BlockUp(d * 8 * 2, d * 8, 4, 2, 1, drop=False)
        self.deconv4 = BlockUp(d * 8 * 2, d * 4, 4, 2, 1, drop=False)
        self.deconv5 = BlockUp(d * 4 * 2, d * 2, 4, 2, 1, drop=False)
        self.deconv6 = BlockUp(d * 2 * 2, d, 4, 2, 1, drop=False)
        self.deconv7 = BlockUp(d * 2, out_ch, 4, 2, 1, norm=False, drop=False)
        # self.out_func = nn.Tanh()
        self.out_func = nn.Sigmoid()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
            
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)

        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)
        e7 = self.conv7(e6)

        d1 = self.deconv1(e7)
        d2 = self.deconv2(d1, e6)
        d3 = self.deconv3(d2, e5)
        d4 = self.deconv4(d3, e4)
        d5 = self.deconv5(d4, e3)
        d6 = self.deconv6(d5, e2)
        d7 = self.deconv7(d6, e1)
        out = self.out_func(d7)

        return out


class HideNet2(nn.Module):
    # initializers
    def __init__(self, in_ch=3, out_ch=3, d=64):
        super(HideNet2, self).__init__()
        # Unet encoder A
        self.enc_a1 = BlockDown(in_ch, d, 4, 2, 1, relu=False, norm=False)
        self.enc_a2 = BlockDown(d, d * 2, 4, 2, 1)
        self.enc_a3 = BlockDown(d * 2, d * 4, 4, 2, 1)
        self.enc_a4 = BlockDown(d * 4, d * 8, 4, 2, 1)
        self.enc_a5 = BlockDown(d * 8, d * 8, 4, 2, 1)
        self.enc_a6 = BlockDown(d * 8, d * 8, 4, 2, 1)
        # self.enc_a7 = BlockDown(d * 8, d * 8, 4, 2, 1, norm=False)
        self.enc_a7 = BlockDown(d * 8, d * 8, 4, 2, 1)

        # Unet encoder B
        self.enc_b1 = BlockDown(in_ch, d, 4, 2, 1, relu=False, norm=False)
        self.enc_b2 = BlockDown(d, d * 2, 4, 2, 1)
        self.enc_b3 = BlockDown(d * 2, d * 4, 4, 2, 1)
        self.enc_b4 = BlockDown(d * 4, d * 8, 4, 2, 1)
        self.enc_b5 = BlockDown(d * 8, d * 8, 4, 2, 1)
        self.enc_b6 = BlockDown(d * 8, d * 8, 4, 2, 1)
        # self.enc_b7 = BlockDown(d * 8, d * 8, 4, 2, 1, norm=False)
        self.enc_b7 = BlockDown(d * 8, d * 8, 4, 2, 1)

        # Unet bottleneck
        self.center = BlockCenter(d * 8 * 2, d * 8, 3, 1, 1, norm=False)

        # Unet decoder
        self.dec_1 = BlockUp(d * 8, d * 8, 4, 2, 1)
        self.dec_2 = BlockUp(d * 8 * 3, d * 8, 4, 2, 1)
        self.dec_3 = BlockUp(d * 8 * 3, d * 8, 4, 2, 1)
        self.dec_4 = BlockUp(d * 8 * 3, d * 4, 4, 2, 1, drop=False)
        self.dec_5 = BlockUp(d * 4 * 3, d * 2, 4, 2, 1, drop=False)
        self.dec_6 = BlockUp(d * 2 * 3, d, 4, 2, 1, drop=False)
        self.dec_7 = BlockUp(d * 3, out_ch, 4, 2, 1, norm=False, drop=False)
        # self.out_func = nn.Tanh()
        self.out_func = nn.Sigmoid()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x1, x2):
        ea1 = self.enc_a1(x1)
        ea2 = self.enc_a2(ea1)
        ea3 = self.enc_a3(ea2)
        ea4 = self.enc_a4(ea3)
        ea5 = self.enc_a5(ea4)
        ea6 = self.enc_a6(ea5)
        ea7 = self.enc_a7(ea6)

        eb1 = self.enc_b1(x2)
        eb2 = self.enc_b2(eb1)
        eb3 = self.enc_b3(eb2)
        eb4 = self.enc_b4(eb3)
        eb5 = self.enc_b5(eb4)
        eb6 = self.enc_b6(eb5)
        eb7 = self.enc_b7(eb6)

        # d1 = self.dec_1(ea7, eb7)  # self.dec_1 = BlockUp(d * 8 * 2, d * 8, 4, 2, 1)
        cen = self.center(ea7, eb7)

        d1 = self.dec_1(cen)
        d2 = self.dec_2(d1, ea6, eb6)
        d3 = self.dec_3(d2, ea5, eb5)
        d4 = self.dec_4(d3, ea4, eb4)
        d5 = self.dec_5(d4, ea3, eb3)
        d6 = self.dec_6(d5, ea2, eb2)
        d7 = self.dec_7(d6, ea1, eb1)
        out = self.out_func(d7)

        return out
    

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class RevealNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, d=64, output_function=nn.Sigmoid):
        super(RevealNet, self).__init__()
        # Unet encoder
        self.conv1 = BlockDown(in_ch, d, 4, 2, 1, relu=False, norm=False)
        self.conv2 = BlockDown(d, d * 2, 4, 2, 1)
        self.conv3 = BlockDown(d * 2, d * 4, 4, 2, 1)
        self.conv4 = BlockDown(d * 4, d * 8, 4, 2, 1)
        self.conv5 = BlockDown(d * 8, d * 8, 4, 2, 1)
        self.conv6 = BlockDown(d * 8, d * 8, 4, 2, 1)
        self.conv7 = BlockDown(d * 8, d * 8, 4, 2, 1, norm=False)

        # Unet decoder
        self.deconv1 = BlockUp(d * 8, d * 8, 4, 2, 1)
        self.deconv2 = BlockUp(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv3 = BlockUp(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv4 = BlockUp(d * 8 * 2, d * 4, 4, 2, 1, drop=False)
        self.deconv5 = BlockUp(d * 4 * 2, d * 2, 4, 2, 1, drop=False)
        self.deconv6 = BlockUp(d * 2 * 2, d, 4, 2, 1, drop=False)
        self.deconv7 = BlockUp(d * 2, out_ch, 4, 2, 1, norm=False, drop=False)
        # self.out_func = nn.Tanh()
        self.out_func = output_function()

    def forward(self, x):
        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)
        e7 = self.conv7(e6)

        d1 = self.deconv1(e7)
        d2 = self.deconv2(d1, e6)
        d3 = self.deconv3(d2, e5)
        d4 = self.deconv4(d3, e4)
        d5 = self.deconv5(d4, e3)
        d6 = self.deconv6(d5, e2)
        d7 = self.deconv7(d6, e1)
        out = self.out_func(d7)

        return out
    

if __name__ == '__main__':
    from torchinfo import summary
    from utils import print_network
    # net = HideNet1()
    net = HideNet2()
    print_network(net)
    summary(net, [(32, 3, 256, 256), (32, 3, 256, 256)], device='cpu')
