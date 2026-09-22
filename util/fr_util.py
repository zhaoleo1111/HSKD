import torch
import numpy as np


def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize / 2) ** 2 + (j - imageSize / 2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0

def mask_radial(img, r):
    rows, cols = img.shape
    mask = torch.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask.cuda()

def generate_high(Images, r):
    # Image: bsxcxhxw, input batched images
    # r: int, radius
    mask = mask_radial(torch.zeros([Images.shape[2], Images.shape[3]]), r)
    bs, c, h, w = Images.shape
    x = Images.reshape([bs * c, h, w])
    fd = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)))
    mask = mask.unsqueeze(0).repeat([bs * c, 1, 1])
    fd = fd * (1.-mask)
    fd = torch.fft.ifftn(torch.fft.ifftshift(fd), dim=(-2, -1))
    fd = torch.real(fd)
    fd = fd.reshape([bs, c, h, w])
    return fd

def generate_low(Images, r):
    # r: int, radius
    mask = mask_radial(torch.zeros([Images.shape[2], Images.shape[3]]), r)
    bs, c, h, w = Images.shape
    x = Images.reshape([bs * c, h, w])
    fd = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)))
    mask = mask.unsqueeze(0).repeat([bs * c, 1, 1])
    fd = fd * mask
    fd = torch.fft.ifftn(torch.fft.ifftshift(fd), dim=(-2, -1))
    fd = torch.real(fd)
    fd = fd.reshape([bs, c, h, w])
    return fd

############################
def make_radial(img,radius=8):
    bs, c, h, w  = img.shape
    low_pass_filter = torch.ones((3, h, w))
    for i in range(h):
        for j in range(w):
            if (i - (h - 1) / 2)**2 + (j -(w - 1) / 2)**2 > radius**2:
                low_pass_filter[:, i, j] = 0
    return low_pass_filter.cuda()

def LowFreqTargetGenerator(imgs):
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1,1).to(imgs.device)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1,1).to(imgs.device)
    low_pass_filter = make_radial(imgs)
    # recover the image to the pre-normalized form
    imgs = imgs * std + mean
    freq_imgs = torch.fft.fft2(imgs)
    freq_imgs = torch.fft.fftshift(freq_imgs, dim=(-2, -1))
    # low pass images
    low_pass_imgs = freq_imgs * low_pass_filter
    low_pass_imgs = torch.fft.ifft2(low_pass_imgs)
    low_pass_imgs = torch.abs(low_pass_imgs)
    low_pass_imgs = (low_pass_imgs - mean) / std
    return low_pass_imgs

def generate_freq(Images, r=8, sample_ratio=0.5):
    # r: int, radius
    mask = mask_radial(torch.zeros([Images.shape[2], Images.shape[3]]), r)
    bs, c, h, w = Images.shape
    x = Images.reshape([bs * c, h, w])
    fd = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)))
    mask = mask.unsqueeze(0).repeat([bs * c, 1, 1])
    rnd = torch.bernoulli(torch.tensor(sample_ratio, dtype=torch.float)).item()
    if rnd == 0:  # high-pass
        mask =  1 - mask
    elif rnd == 1:  # low-pass
        mask = mask
    fd = fd * mask
    fd = torch.fft.ifftn(torch.fft.ifftshift(fd), dim=(-2, -1))
    fd = torch.real(fd)
    fd = fd.reshape([bs, c, h, w])
    return fd

def gen_gaussian_noise(image,SNR):
    """
    :param image: source image
    :param SNR: signal-noise ratio
    :return: noise
    """
    assert len(image.shape) == 3
    H, W, C = image.shape
    noise=torch.random.randn(H, W, 1)
    noise = noise - torch.mean(noise)
    image_power=1/(H*W*3)*np.sum(np.power(image,2))
    noise_variance=image_power/np.power(10,(SNR/10))
    noise=(np.sqrt(noise_variance)/np.std(noise))*noise
    return noise

def mmd_two_distribution(source, target, sigmas):
    sigmas = torch.tensor(sigmas).cuda()
    xy = rbf_kernel(source, target, sigmas)
    xx = rbf_kernel(source, source, sigmas)
    yy = rbf_kernel(target, target, sigmas)
    return xx + yy - 2 * xy

def rbf_kernel(x, y, sigmas):
    sigmas = sigmas.reshape(sigmas.shape + (1,))
    beta = 1. / (2. * sigmas)
    dist = compute_pairwise_distances(x, y)
    dot = -torch.matmul(beta, torch.reshape(dist, (1, -1)))
    exp = torch.mean(torch.exp(dot))
    return exp

def compute_pairwise_distances(x, y):
    dist = torch.zeros(x.size(0),y.size(0)).cuda()
    for i in range(x.size(0)):
        dist[i,:] = torch.sum(torch.square(x[i].expand(y.shape) - y),dim=1)
    return dist        