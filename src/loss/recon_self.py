import torch
import torch.nn as nn
import torch.nn.functional as F

from . import regist_loss


eps = 1e-6

# ============================ #
#  Self-reconstruction loss    #
# ============================ #

@regist_loss
class self_L1():
    def __call__(self, input_data, model_output, data, module):
        img_pd_bsn = model_output['recon'][1]
        with torch.no_grad():
            b2b_img=model_output['recon'][0]
        #
        #    fxw= model_output['recon'][0]
        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']
        loss1=F.l1_loss(img_pd_bsn, target_noisy)
        # loss2=F.l1_loss(fxw, target_noisy)
        loss2=F.l1_loss(b2b_img, target_noisy)

        # weight=output+2*fx
        loss3=F.l1_loss(img_pd_bsn, b2b_img)

        return loss1+loss3+loss2
# class self_L1():
#     def __call__(self, input_data, model_output, data, module):
#         pd2_img_denoised =model_output['recon']
#         target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']
#         loss1=F.l1_loss(pd2_img_denoised, target_noisy)
#
#         return loss1

#
@regist_loss
class self_L2():
    def __call__(self, input_data, model_output, data, module):
        output = model_output['recon']
        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']

        return F.mse_loss(output, target_noisy)
