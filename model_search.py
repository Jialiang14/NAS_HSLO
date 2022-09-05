import math
import torch
import torch.nn as nn
from utils import drop_path
import torch.nn.functional as F
from utils import *
from torchstat import stat
class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):

        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3),
                              stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, size):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, size = size, mode='bilinear', align_corners=True)
        return x

class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x):
        x, skip = x

        # x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.interpolate(x, size=skip.size()[-2:], mode='bilinear', align_corners=True)
        skip = self.skip_conv(skip)

        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        self.blocks = [
            Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))
        ]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                self.blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.blocks_name = []
        for i, block in enumerate(self.blocks):
            self.add_module("Block_{}".format(i), block)
            self.blocks_name.append("Block_{}".format(i))

        # self.block = nn.Sequential(*self.blocks)

    def forward(self, x, sizes = []):
        for i, block_name in enumerate(self.blocks_name):
            x = getattr(self, block_name)(x, sizes[i])
        return x

class Inverted_Bottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, shadow_bn, stride, activation=nn.ReLU6):
        super(Inverted_Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.shadow_bn = shadow_bn
        self.stride = stride
        self.kernel_list = [3, 5, 7, 9]
        self.expansion_rate = [3, 6]
        self.activation = activation(inplace=True)

        self.pw = nn.ModuleList([])
        self.mix_conv = nn.ModuleList([])
        self.mix_bn = nn.ModuleList([])
        self.pw_linear = nn.ModuleList([])

        for t in self.expansion_rate:
            # pw
            self.pw.append(nn.Sequential(
                nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False),
                nn.BatchNorm2d(inplanes * t),
                activation(inplace=True)
            ))
            # dw
            conv_list = nn.ModuleList([])
            for j in self.kernel_list:
                conv_list.append(nn.Sequential(
                    nn.Conv2d(inplanes * t, inplanes * t, kernel_size=j, stride=stride, padding=j // 2,
                              bias=False, groups=inplanes * t),
                    nn.BatchNorm2d(inplanes * t),
                    activation(inplace=True)
                ))

            self.mix_conv.append(conv_list)
            del conv_list
            # pw
            self.pw_linear.append(nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False))

            bn_list = nn.ModuleList([])
            if self.shadow_bn:
                for j in range(len(self.kernel_list)):
                    bn_list.append(nn.BatchNorm2d(outplanes))
                self.mix_bn.append(bn_list)
            else:
                self.mix_bn.append(nn.BatchNorm2d(outplanes))
            del bn_list

    def forward(self, x, choice):
        # choice: {'conv', 'rate'}
        conv_ids = choice['conv']  # conv_ids, e.g. [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]
        m_ = len(conv_ids)  # num of selected paths
        rate_id = choice['rate']  # rate_ids, e.g. 0, 1
        assert m_ in [1, 2, 3, 4]
        assert rate_id in [0, 1]
        residual = x
        # pw
        out = self.pw[rate_id](x)
        # dw
        if m_ == 1:
            out = self.mix_conv[rate_id][conv_ids[0]](out)
        else:
            temp = []
            for id in conv_ids:
                temp.append(self.mix_conv[rate_id][id](out))
            out = sum(temp)
        # pw
        out = self.pw_linear[rate_id](out)
        if self.shadow_bn:
            out = self.mix_bn[rate_id][m_ - 1](out)
        else:
            out = self.mix_bn[rate_id](out)

        # residual
        if self.stride == 1 and self.inplanes == self.outplanes:
            out = out + residual
        return out


channel = [32, 48, 48, 96, 96, 96, 192, 192, 192, 256, 256, 320, 320]
last_channel = 1280


class SuperNetwork(nn.Module):
    def __init__(self, shadow_bn, layers=12, classes=10,final_upsampling=2):
        super(SuperNetwork, self).__init__()
        self.layers = layers
        self.final_upsampling = final_upsampling
        self.stem = nn.Sequential(
            nn.Conv2d(1, channel[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU6(inplace=True)
        )

        self.Inverted_Block = nn.ModuleList([])
        for i in range(self.layers):
            if i in [int(self.layers / 5), int(self.layers * 2 / 5), int(self.layers * 3 / 5), int(self.layers * 4 / 5)]:
                self.Inverted_Block.append(Inverted_Bottleneck(channel[i], channel[i + 1], shadow_bn, stride=2))
            else:
                self.Inverted_Block.append(Inverted_Bottleneck(channel[i], channel[i + 1], shadow_bn, stride=1))

        self.conv1 = nn.Conv2d(320, 256, kernel_size=(1, 1))
        self.p4 = FPNBlock(256, 256)
        self.p3 = FPNBlock(256, 192)
        self.p2 = FPNBlock(256, 96)

        self.s5 = SegmentationBlock(256, 128, n_upsamples=4)
        self.s4 = SegmentationBlock(256, 128, n_upsamples=3)
        self.s3 = SegmentationBlock(256, 128, n_upsamples=2)
        self.s2 = SegmentationBlock(256, 128, n_upsamples=1)

        self.dropout = nn.Dropout2d(p=0.5, inplace=True)
        self.final_conv = nn.Conv2d(272, 1, kernel_size=1, padding=0)
        self.drop_path_prob = 0.2

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0)  # fan-out
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()

    def forward(self, x, choice=True):
        x = self.stem(x)
        for i in range(self.layers):
            x = self.Inverted_Block[i](x, choice[i])
            y = int(self.layers / 5) - 1
            if i == int(self.layers/ 5) - 1:
                c0 = x  # 50
            if i == int(self.layers * 1 / 5):
                c1 = x  # 25
            if i == int(self.layers * 2 / 5):
                c2 = x  # 13
            if i == int(self.layers * 3 / 5):
                c3 = x  # 7
            if i == int(self.layers * 4 / 5):
                c4 = x  # 4
        c5 = x  # 2
        p5 = self.conv1(c5)  # 2
        p4 = self.p4([p5, c4])  # 4 c4为13，插值有参数保持一致
        p3 = self.p3([p4, c3])  # 7
        p2 = self.p2([p3, c2])  # 13

        s5 = self.s5(p5, sizes=[c4.size()[-2:], c3.size()[-2:], c2.size()[-2:], c1.size()[-2:]])  # 13
        s4 = self.s4(p4, sizes=[c3.size()[-2:], c2.size()[-2:], c1.size()[-2:]])
        s3 = self.s3(p3, sizes=[c2.size()[-2:], c1.size()[-2:]])
        s2 = self.s2(p2, sizes=[c1.size()[-2:]])
        x = s5 + s4 + s3 + s2  # 25
        x = torch.cat((x, c1), dim=1)
        x = self.dropout(x)
        #  上采样变成2倍
        if self.final_upsampling is not None and self.final_upsampling > 1:
            x = F.interpolate(x, scale_factor=self.final_upsampling, mode='bilinear', align_corners=False)
        x = torch.cat((x, c0), dim=1)
        x = self.final_conv(x)
        return x


if __name__ == '__main__':
    choice = {
        0: {'conv': [0, 0], 'rate': 1},
        1: {'conv': [0, 0], 'rate': 1},
        2: {'conv': [0, 0], 'rate': 1},
        3: {'conv': [0, 0], 'rate': 1},
        4: {'conv': [0, 0], 'rate': 1},
        5: {'conv': [0, 0], 'rate': 1},
        6: {'conv': [0, 0], 'rate': 1},
        7: {'conv': [0, 0], 'rate': 1},
        8: {'conv': [0, 0], 'rate': 1},
        9: {'conv': [0, 0], 'rate': 1},
        10: {'conv': [0, 0], 'rate': 1},
        11: {'conv': [0, 0], 'rate': 1}}
    temp = choice[0]
    print(temp)
    print(temp['conv'])
    model = SuperNetwork(shadow_bn=False, layers=12, classes=10,final_upsampling=2)
    print(model)
    input = torch.randn(1,1, 50, 50)
    # print(count_parameters_in_MB(model(input, choice)))
    # stat(model,input)
    y = model(input,choice)
    print(y.shape)