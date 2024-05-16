import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional
import cv2
# from ..util.util2 import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling
# from ..util.util2 import randomArrangement, inverseRandomArrangement, showTensor
from ..util import util2
from . import regist_model
from torchvision import transforms
from .DBSNl import DBSNl
import numpy as np
from .Mask import Masker,depth_to_space,space_to_depth
#迭代
np.set_printoptions(threshold=np.sys.maxsize)
@regist_model
class APBSN(nn.Module):
    '''
    Asymmetric PD Blind-Spot Network (AP-BSN)
    '''

    def __init__(self, pd_a=5, pd_b=2, pd_pad=2, R3=False, R3_T=8, R3_p=0.16,
                 bsn='DBSNl', in_ch=3, bsn_base_ch=128, bsn_num_module=9
):
        '''
        Args:
            pd_a           : 'PD stride factor' during training
            pd_b           : 'PD stride factor' during inference
            pd_pad         : pad size between sub-images by PD process
            R3             : flag of 'Random Replacing Refinement'
            R3_T           : number of masks for R3
            R3_p           : probability of R3
            bsn            : blind-spot network type
            in_ch          : number of input image channel
            bsn_base_ch    : number of bsn base channel
            bsn_num_module : number of module
        '''
        super().__init__()

        # network hyper-parameters
        self.pd_a = pd_a
        self.pd_b = pd_b
        self.pd_pad = pd_pad
        self.R3 = R3
        self.R3_T = R3_T
        self.R3_p = R3_p


        # self.attenbsn=ABSNl(in_ch, in_ch, bsn_base_ch, bsn_num_module)
        # self.cnn_atenNet = cnn_atenNet(in_ch, width,f_scale,ss_exp_factor,stride)
        # self.plainNet=plainNet(stride, in_ch,reduction)# cnn
        # define network
        if bsn == 'DBSNl':
            self.bsn = DBSNl(in_ch, in_ch, bsn_base_ch, bsn_num_module)
        else:
            raise NotImplementedError('bsn %s is not implemented' % bsn)
        ly = []
        ly += [nn.Conv2d(12, 3, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        self.conv1 = nn.Sequential(*ly)
        ly = []
        ly += [nn.Conv2d(3, 12, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        self.conv2 = nn.Sequential(*ly)
        # self.conv = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1,bias=False)
        # self.maskconv=CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)
        self.masker = Masker(width=4, mode='interpolate', mask_type='all')
# forward_ra

    def forward(self, img, pd=None):
        '''
        Foward function includes sequence of PD, BSN and inverse PD processes.
        Note that denoise() function is used during inference time (for differenct pd factor and R3).
        '''
        # default pd factor is training factor (a)
        if pd is None: pd = self.pd_a
        b,c,h,w=img.shape

        # pad images for PD process
        if h % pd != 0:
            img = F.pad(img, (0, 0, 0, pd - h % pd), mode='constant', value=0)
        if w % pd != 0:
            img = F.pad(img, (0, pd - w % pd, 0, 0), mode='constant', value=0)

        # do PD

        pd_img = util2.pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)

        # random rearrangement xiaoyu 2023/5/23
        pd_img, random2seq = util2.randomArrangement(pd_img, pd)

        pd_img_denoised = self.bsn(pd_img,is_masked=True)

        # inverse rearrangement xiaoyu 2023/5/23
        pd_img_denoised = util2.inverseRandomArrangement(pd_img_denoised, random2seq, pd)

        # do inverse PD

        img_pd_bsn = util2.pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
        n, c, h, w = img.shape

        x2, mask = self.masker.train(img)

        x2 = self.bsn(x2, is_masked=False)
        ##b2s恢复
        b2b_img = (x2 * mask).view(n, -1, c, h, w).sum(dim=1)
        # n, c, h, w = maskimg.shape
        # if h % 3 != 0:
        #     maskimg = F.pad(maskimg, (0, 0, 0, 3 - h % 3), mode='constant', value=0)
        # if w % 3 != 0:
        #     maskimg = F.pad(maskimg, (0, 3 - w % 3, 0, 0), mode='constant', value=0)
        # x2, mask = self.masker.train(maskimg)
        #
        # x2 = self.bsn(x2,is_masked=False)
        # ##b2s恢复
        # b2b_img = (x2 * mask).view(n, -1, c, h, w).sum(dim=1)
        return b2b_img,img_pd_bsn
#img1_pd_bsn=self.forward(img)

#


    def forward_mpd(self, img, pd=None):
        '''
        Foward function includes sequence of PD, BSN and inverse PD processes.
        Note that denoise() function is used during inference time (for differenct pd factor and R3).
        '''
        # default pd factor is training factor (a)a
        if pd is None: pd = self.pd_a

        b, c, h, w = img.shape

        # pad images for PD process
        if h % pd != 0:
            img = F.pad(img, (0, 0, 0, pd - h % pd), mode='constant', value=0)
        if w % self.pd_b != 0:
            img = F.pad(img, (0, pd - w % pd, 0, 0), mode='constant', value=0)

        # do PD
        if pd > 1:
            pd_img = util2.pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            pd_img = F.pad(img, (p, p, p, p))

        img_pd, random2seq = util2.randomArrangement(pd_img, pd)

        # forward blind-spot network
        pd_img_denoised = self.bsn(img_pd,is_masked=True)

        # inverse rearrangement xiaoyu 2023/5/23
        pd_img_denoised = util2.inverseRandomArrangement(pd_img_denoised, random2seq, pd)

        # do inverse PD
        if pd > 1:
            img_pd_bsn = util2.pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            img_pd_bsn = pd_img_denoised[:, :, p:-p, p:-p]

        return img_pd_bsn

# class CentralMaskedConv2d(nn.Conv2d):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         self.register_buffer('mask', self.weight.data.clone())
#         _, _, kH, kW = self.weight.size()
#         self.mask.fill_(1)
#         self.mask[:, :, kH // 2, kH // 2] = 0
#
#     def forward(self, x):
#         self.weight.data *= self.mask
#         return super().forward(x)
#     def denoise(self, x):
#         '''
#         Denoising process for inference.
#         '''
#         b, c, h, w = x.shape
#
#         # pad images for PD process
#         if h % self.pd_b != 0:
#             x = F.pad(x, (0, 0, 0, self.pd_b - h % self.pd_b), mode='constant', value=0)
#         if w % self.pd_b != 0:
#             x = F.pad(x, (0, self.pd_b - w % self.pd_b, 0, 0), mode='constant', value=0)
#
#         # forward PD-BSN process with inference pd factor
#         img_pd_bsn = self.forward_mpd(img=x, pd=self.pd_b)
#         return img_pd_bsn

        # Random Replacing Refinement
        # if not self.R3:
        #     ''' Directly return the result (w/o R3) '''
        #     return img_pd_bsn[:, :, :h, :w]
        # else:
        #     denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
        #     for t in range(self.R3_T):
        #         indice = torch.rand_like(x)
        #         mask = indice < self.R3_p
        #
        #         tmp_input = torch.clone(img_pd_bsn).detach()
        #         tmp_input[mask] = x[mask]
        #         p = self.pd_pad
        #         tmp_input = F.pad(tmp_input, (p, p, p, p), mode='reflect')
        #         if self.pd_pad == 0:
        #             denoised[..., t] = self.bsn(tmp_input)
        #         else:
        #             denoised[..., t] = self.bsn(tmp_input)[:, :, p:-p, p:-p]
        #
        #     return torch.mean(denoised, dim=-1)
    def denoise(self, x):
        '''
        Denoising process for inference.
        '''

        b, c, h, w = x.shape


        # ============== PD = 2 ====================
        img_pd2_bsn = x
        if h % self.pd_b != 0:
            img_pd2_bsn = F.pad(img_pd2_bsn, (0, 0, 0, self.pd_b - h % self.pd_b), mode='constant', value=0)
        if w % self.pd_b != 0:
            img_pd2_bsn = F.pad(img_pd2_bsn, (0, self.pd_b - w % self.pd_b, 0, 0), mode='constant', value=0)
        img_pd2_bsn = self.forward_mpd(img_pd2_bsn, pd=2)

        # # ============== PD = 5 ====================
        img_pd5_bsn = x
        if h % self.pd_a != 0:
            img_pd5_bsn = F.pad(img_pd5_bsn, (0, 0, 0, self.pd_a - h % self.pd_a), mode='constant', value=0)
        if w % self.pd_a != 0:
            img_pd5_bsn = F.pad(img_pd5_bsn, (0, self.pd_a - w % self.pd_a, 0, 0), mode='constant', value=0)
        img_pd5_bsn = self.forward_mpd(img_pd5_bsn, pd=5)
        img_pd5_bsn = img_pd5_bsn[:, :, :h, :w]
        # ============== FUSE 1 ====================
        img_pd1_bsn=self.bsn(x,is_masked=True)
        img_pd_bsn = torch.add(torch.mul(img_pd5_bsn, 0.7), torch.mul(img_pd1_bsn, 0.3))#鍘? 9锛?1

        # ============== FUSE 2 ====================
        img_pd_bsn = torch.add(torch.mul(img_pd_bsn, 0.2), torch.mul(img_pd2_bsn, 0.8))
        # return img_pd_bsn[:, :, :h, :w]

        # Random Replacing Refinement

        if not self.R3:
        #     ''' Directly return the result (w/o R3) '''
            return img_pd2_bsn[:, :, :h, :w]
        else:
            # x = fx
            denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
            for t in range(self.R3_T):
                indice = torch.rand_like(x)
                mask = indice < self.R3_p

                tmp_input = torch.clone(img_pd_bsn).detach()
                tmp_input[mask] = x[mask]
                p = self.pd_pad
                tmp_input = F.pad(tmp_input, (p, p, p, p), mode='reflect')
                if self.pd_pad == 0:
                    denoised[..., t] = self.bsn(tmp_input,is_masked=True)
                else:
                    denoised[..., t] = self.bsn(tmp_input,is_masked=True)[:, :, p:-p, p:-p]

            rimg_pd_bsn = torch.mean(denoised, dim=-1)

        return rimg_pd_bsn[:, :, :h, :w]