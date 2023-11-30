#!/usr/bin/python
# -*- encoding: utf-8 -*-
#!/usr/bin/python
# -*- encoding: utf-8 -*-
# ------------------------------------------------------------------------------
# Written by Kaige Li (kglee1994@163.com)
# ------------------------------------------------------------------------------

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.base_models.resnetv1c import resnet18

from torch.nn import BatchNorm2d

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.keras_init_weight()
    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d,nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                # nn.init.xavier_normal_(ly.weight,gain=nn.init.calculate_gain('relu'))
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x




class Backbone(nn.Module):
    def __init__(self, backbone='resnet18', pretrained_base=True, norm_layer=nn.BatchNorm2d):
        super(Backbone, self).__init__()
        if backbone == 'resnet18':
            pretrained = resnet18(pretrained=pretrained_base)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.conv1 = pretrained.conv1
        self.bn1  = pretrained.bn1
        self.maxpool = pretrained.maxpool
        self.layer1 = pretrained.layer1
        self.layer2 = pretrained.layer2
        self.layer3 = pretrained.layer3
        self.layer4 = pretrained.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x,True)
        x = self.maxpool(x)
        x = self.layer1(x)
        c2 = self.layer2(x)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c2,  c3, c4


class BiSeNetOutput(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.up_factor = up_factor
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=True)
        self.keras_init_weight()
    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = F.interpolate(x, scale_factor=self.up_factor, mode='bilinear', align_corners=True)
        return x
    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.xavier_normal_(ly.weight)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class GeneralizedMeanPoolingBase(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingBase, self).__init__()
        assert norm > 0
        self.p = float(norm)  # TODO  固定 p
        # self.p = nn.Parameter(torch.ones(1) * norm)
        self.output_size = output_size
        self.eps = eps
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        return (self.avg_pool(x ** self.p) + 1e-12) ** (1 / self.p)



class SCE(nn.Module):
    def __init__(self, in_channels = 512, out_channels= 128 , grids=(6, 3, 2, 1)):
        super(SCE, self).__init__()

        self.reduce_channel = ConvBNReLU(in_channels, out_channels,1,1,0)
        self.grids = grids
        print('grid ar near',self.grids)
        self.spp = nn.Sequential()
        self.spp.add_module('spp_1', ConvBNReLU(out_channels, out_channels,1,1,0))
        self.spp.add_module('spp_2', ConvBNReLU(out_channels, out_channels,1,1,0))
        self.spp.add_module('spp_3', ConvBNReLU(out_channels, out_channels,1,1,0))
        self.spp.add_module('spp_4', ConvBNReLU(out_channels, out_channels,1,1,0))

        self.upsampling_method = lambda x, size: F.interpolate(x, size, mode='nearest')

        self.spatial_attention = nn.Sequential(
            ConvBNReLU(out_channels * 4, out_channels, 1, 1, 0),
            nn.Conv2d(out_channels, 4, kernel_size=1, bias=False), ##
            nn.Sigmoid()
        )
        self.keras_init_weight()
        self.spatial_attention[1].weight.data.zero_()


    def forward(self, x):

        size = x.size()[2:]

        ar = size[1] / size[0]
        x = self.reduce_channel(x) # 降维

        context = []
        for i in range(len(self.grids)):
            grid_size = (self.grids[i], max(1, round(ar * self.grids[i])))
            # grid_size = (self.grids[i], self.grids[i])
            x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            x_pooled = self.spp[i].forward(x_pooled)
            x_pooled = self.upsampling_method(x_pooled,size)
            context.append(x_pooled)
            # out = out + x_pooled

        spatial_att = self.spatial_attention(torch.cat(context,dim=1))  + 1 ## truple 4

        x = x + context[0] * spatial_att[:, 0:1, :, :] + context[1] * spatial_att[:, 1:2, :, :]  \
            + context[2] * spatial_att[:, 2:3, :, :] + context[3] * spatial_att[:, 3:4, :, :]


        return x
    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.xavier_normal_(ly.weight)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class SFF(nn.Module):
    """FFM"""

    def __init__(self, low_channels = 128, high_channels = 128, out_channels = 256, norm_layer=nn.BatchNorm2d, **kwargs):
        super(SFF, self).__init__()
        print('SFF+ att sum + layernorm')


        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 3,1,1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 3,1,1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

        # self.avg_pool = nn.AdaptiveMaxPool2d(1)
        # self.avg_pool =  nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool = GeneralizedMeanPoolingBase(norm=3)
        k_size = 5
        self.conv_1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv_2 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)


        self.D = out_channels

        self.init_weight()
        # self.keras_init_weight()
        # self.conv_2.weight.data.zero_()

    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d,nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x_low, x_high):

        b,_,h,w = x_high.size()
        x_low = self.conv_low(x_low)
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        # x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='nearest')

        x_high = self.conv_high(x_high)

        d  = torch.cat([self.avg_pool(x_low).unsqueeze(1), self.avg_pool(x_high).unsqueeze(1)],dim=1)
        d = d.transpose(1, 2).flatten(1, 2) # B 2*C  1  1

        # 生成的权重 是 高低交叉的。
        d = self.conv_1(d.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) # B 2C 1 1

        d = self.conv_2(d.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) # B 2C 1 1

        d = d.reshape(b, self.D, 2 , 1, 1).transpose(1, 2).transpose(0, 1) # 2 B C 1  1

        # d = 1 + torch.tanh(d)
        d = torch.sigmoid(d) # TODO

        x_fuse = d[0] * x_low + d[1] * x_high

        return x_fuse

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)




class SANet(nn.Module):

    def __init__(self, n_classes=19, output_aux=True, *args, **kwargs):
        super(SANet, self).__init__()
        self.resnet = Backbone()

        self.sce = SCE()

        self.sff1 = SFF(512,256,128)
        self.sff2 = SFF(128,128,128)


        self.conv_out32 = BiSeNetOutput(128, 128, n_classes, up_factor=8)

        self.output_aux = output_aux
        if self.output_aux:
            self.conv_out16 = BiSeNetOutput(256, 64, n_classes, up_factor=16)

    def forward(self, x):
        # 128
        feat8,dsn,feat32 = self.resnet(x)

        f = self.sce(feat32)

        f = self.sff1f(f,dsn)
        f = self.sff2(f,feat8)

        f = self.conv_out32(f)

        if self.output_aux:
            aux1 = self.conv_out16(dsn)
            return f, aux1

        f = f.argmax(dim=1)
        return f


def get_seg_model(cfg, imgnet_pretrained=True):
    model = SANet(num_classes=cfg.DATASET.NUM_CLASSES, output_aux=True)

    return model


def get_pred_model(name, num_classes):
    model = SANet(num_classes=num_classes, output_aux=False)

    return model


if __name__ == '__main__':

    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    device = torch.device('cuda')
    model = get_pred_model(name='SANet', num_classes=19)
    model.eval()
    model.to(device)
    iterations = None

    input = torch.randn(1, 3, 1024, 2048).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)