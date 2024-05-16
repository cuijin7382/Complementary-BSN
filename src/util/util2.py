from math import exp
import torch.nn as nn
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image
import torchvision.transforms as transforms
import lpips
from DISTS_pytorch import DISTS

# from src.model.APBSN import APBSN
from ..model import get_model_class
def np2tensor(n:np.array):
    '''
    transform numpy array (image) to torch Tensor
    BGR -> RGB
    (h,w,c) -> (c,h,w)
    '''
    # gray
    if len(n.shape) == 2:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(n, (2,0,1))))
    # RGB -> BGR
    elif len(n.shape) == 3:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(np.flip(n, axis=2), (2,0,1))))
    else:
        raise RuntimeError('wrong numpy dimensions : %s'%(n.shape,))

def tensor2np(t:torch.Tensor):
    '''
    transform torch Tensor to numpy having opencv image form.
    RGB -> BGR
    (c,h,w) -> (h,w,c)
    '''
    t = t.cpu().detach()

    # gray
    if len(t.shape) == 2:
        return t.permute(1,2,0).numpy()
    # RGB -> BGR
    elif len(t.shape) == 3:
        return np.flip(t.permute(1,2,0).numpy(), axis=2)
    # image batch
    elif len(t.shape) == 4:
        return np.flip(t.permute(0,2,3,1).numpy(), axis=3)
    else:
        raise RuntimeError('wrong tensor dimensions : %s'%(t.shape,))

def imwrite_tensor(t, name='test.png'):
    cv2.imwrite('./%s'%name, tensor2np(t.cpu()))

def imread_tensor(name='test'):
    return np2tensor(cv2.imread('./%s'%name))

def rot_hflip_img(img:torch.Tensor, rot_times:int=0, hflip:int=0):
    '''
    rotate '90 x times degree' & horizontal flip image
    (shape of img: b,c,h,w or c,h,w)
    '''
    b=0 if len(img.shape)==3 else 1
    # no flip
    if hflip % 2 == 0:
        # 0 degrees
        if rot_times % 4 == 0:
            return img
        # 90 degrees
        elif rot_times % 4 == 1:
            return img.flip(b+1).transpose(b+1,b+2)
        # 180 degrees
        elif rot_times % 4 == 2:
            return img.flip(b+2).flip(b+1)
        # 270 degrees
        else:
            return img.flip(b+2).transpose(b+1,b+2)
    # horizontal flip
    else:
        # 0 degrees
        if rot_times % 4 == 0:
            return img.flip(b+2)
        # 90 degrees
        elif rot_times % 4 == 1:
            return img.flip(b+1).flip(b+2).transpose(b+1,b+2)
        # 180 degrees
        elif rot_times % 4 == 2:
            return img.flip(b+1)
        # 270 degrees
        else:
            return img.transpose(b+1,b+2)

def pixel_shuffle_down_sampling(x:torch.Tensor, f:int, pad:int=0, pad_value:float=0.):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,3,2,4).reshape(c, w+2*f*pad, h+2*f*pad)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(b,c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,2,4,3,5).reshape(b,c,w+2*f*pad, h+2*f*pad)

def pixel_shuffle_up_sampling(x:torch.Tensor, f:int, pad:int=0):
    '''
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        before_shuffle = x.view(c,f,w//f,f,h//f).permute(0,1,3,2,4).reshape(c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        before_shuffle = x.view(b,c,f,w//f,f,h//f).permute(0,1,2,4,3,5).reshape(b,c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)


def pixel_shuffle_down_sampling_pd(x: torch.Tensor, f: int, pad: int = 0, pad_value: float = 0.):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    '''
    # single image tensor
    if len(x.shape) == 3:
        # c, w, h = x.shape
        # unshuffled = F.pixel_unshuffle(x, f)
        # if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        # return -1
        pass
    # batched image tensor
    else:
        b, c, w, h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), 'reflect')
        unshuffled = unshuffled.view(b, c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 2, 3, 1, 4,
                                                                                             5).contiguous()
        unshuffled = unshuffled.view(-1, c, w // f + 2 * pad, h // f + 2 * pad).contiguous()
        return unshuffled


def pixel_shuffle_up_sampling_pd(x: torch.Tensor, f: int, pad: int = 0):
    '''
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    '''
    # single image tensor
    if len(x.shape) == 3:
        # c, w, h = x.shape
        # before_shuffle = x.view(c, f, w // f, f, h // f).permute(0, 1, 3, 2, 4).reshape(c * f * f, w // f, h // f)
        # if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        # return -1
        pass
    # batched image tensor
    else:
        b, c, w, h = x.shape
        b = b // (f * f)
        before_shuffle = x.view(b, f, f, c, w, h)
        before_shuffle = before_shuffle.permute(0, 3, 1, 2, 4, 5).contiguous()
        before_shuffle = before_shuffle.view(b, c * f * f, w, h)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)
def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

def psnr(img1, img2):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    if len(img1.shape) == 4:
        img1 = img1[0]
    if len(img2.shape) == 4:
        img2 = img2[0]

    # tensor to numpy
    if isinstance(img1, torch.Tensor):
        img1 = tensor2np(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2np(img2)

    # numpy value cliping & chnage type to uint8
    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)

    return peak_signal_noise_ratio(img1, img2, data_range=255)

def lpips2(img1, img2):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    # print(img1.is_cuda, img2.is_cuda)

    device = torch.device("cuda:0")
    # img1=img1.to(device)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = APBSN().to(device)
    lpips_model = lpips.LPIPS(net="vgg").to(device)#alex


    # if len(img1.shape) == 4:
    #     img1 = img1[0]
    # if len(img2.shape) == 4:
    #     img2 = img2[0]

    # # tensor to numpy
    # if isinstance(img1, torch.Tensor):
    #     img1 = tensor2np(img1)
    # if isinstance(img2, torch.Tensor):
    #     img2 = tensor2np(img2)
    #
    # # numpy value cliping & chnage type to uint8
    # img1 = np.clip(img1, 0, 255)
    # img2 = np.clip(img2, 0, 255)
    distance = lpips_model(img1, img2)
    return distance.item()

def dists2(img1, img2):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    # print(img1.is_cuda, img2.is_cuda)

    device = torch.device("cuda:0")
    # img1=img1.to(device)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = APBSN().to(device)
    # dists_model = dists.DISTS().to(device)#alex

    D = DISTS().to(device)
    # calculate DISTS between X, Y (a batch of RGB images, data range: 0~1)
    # X: (N,C,H,W)
    # Y: (N,C,H,W)
    dists_value = D(img1, img2)
    # set 'require_grad=True, batch_average=True' to get a scalar value as loss.
    # dists_loss = D(img1, img2, require_grad=True, batch_average=True)
    # dists_loss.backward()

    # if len(img1.shape) == 4:
    #     img1 = img1[0]
    # if len(img2.shape) == 4:
    #     img2 = img2[0]

    # # tensor to numpy
    # if isinstance(img1, torch.Tensor):
    #     img1 = tensor2np(img1)
    # if isinstance(img2, torch.Tensor):
    #     img2 = tensor2np(img2)
    #
    # # numpy value cliping & chnage type to uint8
    # img1 = np.clip(img1, 0, 255)
    # img2 = np.clip(img2, 0, 255)
    # distance = dists_model(img1, img2)
    return dists_value.item()

def ssim(img1, img2):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    if len(img1.shape) == 4:
        img1 = img1[0]
    if len(img2.shape) == 4:
        img2 = img2[0]

    # tensor to numpy
    if isinstance(img1, torch.Tensor):
        img1 = tensor2np(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2np(img2)

    # numpy value cliping
    img2 = np.clip(img2, 0, 255)
    img1 = np.clip(img1, 0, 255)

    return structural_similarity(img1, img2, multichannel=True, data_range=255)

def get_gaussian_2d_filter(window_size, sigma, channel=1, device=torch.device('cpu')):
    '''
    return 2d gaussian filter window as tensor form
    Arg:
        window_size : filter window size
        sigma : standard deviation
    '''
    gauss = torch.ones(window_size, device=device)
    for x in range(window_size): gauss[x] = exp(-(x - window_size//2)**2/float(2*sigma**2))
    gauss = gauss.unsqueeze(1)
    #gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)], device=device).unsqueeze(1)
    filter2d = gauss.mm(gauss.t()).float()
    filter2d = (filter2d/filter2d.sum()).unsqueeze(0).unsqueeze(0)
    return filter2d.expand(channel, 1, window_size, window_size)

def get_mean_2d_filter(window_size, channel=1, device=torch.device('cpu')):
    '''
    return 2d mean filter as tensor form
    Args:
        window_size : filter window size
    '''
    window = torch.ones((window_size, window_size), device=device)
    window = (window/window.sum()).unsqueeze(0).unsqueeze(0)
    return window.expand(channel, 1, window_size, window_size)

def mean_conv2d(x, window_size=None, window=None, filter_type='gau', sigma=None, keep_sigma=False, padd=True):
    '''
    color channel-wise 2d mean or gaussian convolution
    Args:
        x : input image
        window_size : filter window size
        filter_type(opt) : 'gau' or 'mean'
        sigma : standard deviation of gaussian filter
    '''
    b_x = x.unsqueeze(0) if len(x.shape) == 3 else x

    if window is None:
        if sigma is None: sigma = (window_size-1)/6
        if filter_type == 'gau':
            window = get_gaussian_2d_filter(window_size, sigma=sigma, channel=b_x.shape[1], device=x.device)
        else:
            window = get_mean_2d_filter(window_size, channel=b_x.shape[1], device=x.device)
    else:
        window_size = window.shape[-1]

    if padd:
        pl = (window_size-1)//2
        b_x = F.pad(b_x, (pl,pl,pl,pl), 'reflect')

    m_b_x = F.conv2d(b_x, window, groups=b_x.shape[1])

    if keep_sigma:
        m_b_x /= (window**2).sum().sqrt()

    if len(x.shape) == 4:
        return m_b_x
    elif len(x.shape) == 3:
        return m_b_x.squeeze(0)
    else:
        raise ValueError('input image shape is not correct')




def randomArrangement(x:torch.Tensor, pd_factor:int):
    """
    xiaoyu 2023/5/23\n
    用于将pd降采样结果进一步打乱\n
    :param x: pd降采样结果， 大小为[b,c,h,w]
    :param pd_factor: 表示每行有多少个小图像块
    :return: [1]打乱的结果， [2]用于还原打乱结果的映射
    """
    #np.random.permutation是一个随机排列函数,就是将输入的数据进行随机排列
    seq2random = np.random.permutation(range(0, pd_factor*pd_factor))  # 置乱数组 seq2random[i]表示x第i个块应该放在random_x的哪一块

    random2seq = np.zeros_like(seq2random)  # random2seq[i]表示random_x的第i块应该被放在x的哪一块
    for i in range(0, len(seq2random)):#恢复数组
        random2seq[seq2random[i]] = i

    random_x = torch.zeros_like(x)

    b, c, h, w = x.shape
    assert h % pd_factor == 0, "dim[-2] of input x %d cannot be divided by %d" % (h, pd_factor) #确保能被整除
    assert w % pd_factor == 0, "dim[-1] of input x %d cannot be divided by %d" % (w, pd_factor)
    sub_h = h//pd_factor#h//有多少小块 =子块的高
    sub_w = w//pd_factor
    idx = 0

    for i in range(0, pd_factor):
        for j in range(0, pd_factor):
            random_idx = seq2random[idx]
            random_j = random_idx % pd_factor
            random_i = random_idx // pd_factor
            random_x[:, :, random_i*sub_h:(random_i+1)*sub_h, random_j*sub_w:(random_j+1)*sub_w] = x[:, :, i*sub_h:(i+1)*sub_h, j*sub_w:(j+1)*sub_w]#没看懂
            idx += 1

    return random_x, random2seq


def inverseRandomArrangement(random_x:torch.Tensor, random2seq:np.ndarray, pd_factor:int):
    """
    xiaoyu 2023/5/23 \n
    用于还原被randomArrangement(...)打乱的张量 \n
    :param random_x: 乱序张量
    :param random2seq: 逆映射
    :param pd_factor: 表示每行有多少个小图像块
    :return: [1]顺序正确的张量
    """

    x = torch.zeros_like(random_x)

    b, c, h, w = random_x.shape
    assert h % pd_factor == 0, "dim[-2] of input x %d cannot be divided by %d" % (h, pd_factor)
    assert w % pd_factor == 0, "dim[-1] of input x %d cannot be divided by %d" % (w, pd_factor)
    sub_h = h // pd_factor
    sub_w = w // pd_factor

    random_idx = 0

    for random_i in range(0, pd_factor):
        for random_j in range(0, pd_factor):
            idx = random2seq[random_idx]
            j = idx % pd_factor
            i = idx // pd_factor
            x[:, :, i*sub_h:(i+1)*sub_h, j*sub_w:(j+1)*sub_w] = random_x[:, :, random_i*sub_h:(random_i+1)*sub_h, random_j*sub_w:(random_j+1)*sub_w]
            random_idx += 1

    return x


def showTensor(x:torch.Tensor):
    """
    xiaoyu 2023/5/23 \n
    Show the img and make an interupt, for testing only
    :param x: a tensor of shape [1, c, h, w]
    :return: None
    """

    data = torch.permute(x, (0, 2, 3, 1))
    data = data.cpu().detach().numpy()
    data = np.uint8(data[0])
    print(data.shape)
    cv2.imshow("data", data)
    cv2.waitKey(0)


def std(img, window_size=7): #改img: torch.Tensor 5.26
    """
    xiaoyu 2023/5/25 \n
    修改自 "Spatially Adaptive Self-Supervised Learning for Real-World Image Denoising" (CVPR 2023) \n
    :param img: 一个[b, 1, h, w]张量
    :param window_size: 滑动窗口的大小，这个滑动窗口用来计算方差
    :return: 一个[b, 1, h, w]张量
    """
    assert window_size % 2 == 1 #判断是否符合
    pad = window_size // 2

    # calculate std on the mean image of the color channels
    img = torch.mean(img, dim=1, keepdim=True)
    N, C, H, W = img.shape
    img = torch.nn.functional.pad(img, [pad] * 4, mode='reflect')
    img = torch.nn.functional.unfold(img, kernel_size=window_size)
    img = img.view(N, C, window_size * window_size, H, W)
    img = img - torch.mean(img, dim=2, keepdim=True)
    img = img * img
    img = torch.mean(img, dim=2, keepdim=True)
    img = torch.sqrt(img)
    img = img.squeeze(2)
    #normalization
    img = torch.sub(img, torch.min(img))#减
    assert torch.max(img) > 0.0, "maximun std equals 0"
    img = torch.div(img, torch.max(img))#除
    img = torch.cat([img, img, img], dim=1) #cat三个向量？

    return img


def fuseTensor(ap55: torch.Tensor, ap52: torch.Tensor, weightMap: torch.Tensor = None):
    """
    xiaoyu 2023/5/25 \n
    返回ap55与ap52的按元素加权 \n
    :param ap55: AP5/5 结果, 形状应该为[b, c, w, h]
    :param ap52: AP5/2 结果, 形状应该为[b, c, w, h]
    :param weightMap: element-wise权重
    :return: 一个张量,其形状与ap55或ap52相同
    """
    if weightMap is None:
        w1 = std(torch.add(ap55, ap52),window_size=7)
    else:
        w1 = weightMap
    # w0 = torch.add(torch.neg(weightMap), 1)
    w0 = torch.add(torch.neg(w1), 1)
    #  alpha=generate_alpha(input)
    #  if alpha>0.5:
        #return  w0 ?
    #  else：
    #


    ap55_weighted = torch.mul(ap55, w0)
    ap52_weighted = torch.mul(ap52, w1)
    return torch.add(ap55_weighted, ap52_weighted)

#5.26 判断
def generate_alpha(input, lower=1, upper=5):  #判断纹理or平坦？
    N, C, H, W = input.shape
    ratio = input.new_ones((N, 1, H, W)) * 0.5
    input_std = std(input)
    ratio[input_std < lower] = torch.sigmoid((input_std - lower))[input_std < lower]
    ratio[input_std > upper] = torch.sigmoid((input_std - upper))[input_std > upper]
    ratio = ratio.detach()#alpha


    return ratio

def std_ori(img, window_size=7): #计算方差 输入是图？
    assert window_size % 2 == 1
    pad = window_size // 2

    # calculate std on the mean image of the color channels
    img = torch.mean(img, dim=1, keepdim=True)
    N, C, H, W = img.shape
    img = nn.functional.pad(img, [pad] * 4, mode='reflect')
    img = nn.functional.unfold(img, kernel_size=window_size)
    img = img.view(N, C, window_size * window_size, H, W)
    img = img - torch.mean(img, dim=2, keepdim=True)
    img = img * img
    img = torch.mean(img, dim=2, keepdim=True)
    img = torch.sqrt(img)
    img = img.squeeze(2)
    #img = img.numpy()
    print(img.shape)
    img = img.permute(0, 2, 3, 1)[0]
    # img = torch.permute(img, (0, 2, 3, 1))[0]
    img=img.cpu().numpy()
    img = np.uint8(img)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    return img






def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img

def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image