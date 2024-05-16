import torch
import torch.nn as nn
import torch.nn.functional as F

from . import regist_model

import numpy as np
import cv2
@regist_model
class DBSNl(nn.Module):
    '''
    Dilated Blind-Spot Network (cutomized light version)

    self-implemented version of the network from "Unpaired Learning of Deep Image Denoising (ECCV 2020)"
    and several modificaions are included.
    see our supple for more details.
    '''
    # def __init__(self, in_ch=3, out_ch=3, base_ch=128, num_module=9,group = 1, head_ch = None, br2_blc = 6):
    def __init__(self, in_ch=3, out_ch=3, base_ch=128, num_module=9):
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
        '''
        super().__init__()

        assert base_ch%2 == 0, "base channel should be divided with 2"
        # self.is_mask=is_masked
        ly = []
        ly += [ nn.Conv2d(in_ch, base_ch, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly)
        self.branch1_maskconv=mask_conv(2,base_ch)
        self.branch2_maskconv = mask_conv(3, base_ch)
        self.branch1_nomaskconv = nomask_conv(base_ch)
        self.branch2_nomaskconv = nomask_conv( base_ch)
        self.branch1 = DC_branchl(2, base_ch, num_module)
        self.branch2 = DC_branchl(3, base_ch, num_module)

        ly = []
        ly += [ nn.Conv2d(base_ch*2,  base_ch,    kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch,    base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, out_ch,     kernel_size=1) ]
        self.tail = nn.Sequential(*ly)#重建图像




    def forward(self, x,is_masked):


        x = self.head(x)
        if is_masked:
            br1=self.branch1_maskconv(x)
            br2 = self.branch2_maskconv(x)
        else:
            br1=self.branch1_nomaskconv(x)
            br2=self.branch2_nomaskconv(x)
        br1 = self.branch1(br1)

        br2 = self.branch2(br2)

        x = torch.cat([br1, br2], dim=1)

        #msf
        x=self.tail(x)

        # x=self.catten(x)
        return x

    def _initialize_weights(self):
        # Liyong version
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)

#
class mask_conv(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]

        self.body = nn.Sequential(*ly)
    def forward(self, x):
        return self.body(x)


class nomask_conv(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        ly = []
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        self.body = nn.Sequential(*ly)
    def forward(self, x):
        return self.body(x)

class DC_branchl(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()

        ly = []

        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [DCl(stride, in_ch) for _ in range(num_module)]

        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return self.body(x)

class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride) ]

        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)

class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH//2, kH//2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
