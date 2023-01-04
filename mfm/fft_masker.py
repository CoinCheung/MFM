

import random
import torch


class MaskedFFT:

    def __init__(self, rad=16):
        self.rad = rad
        self.mask_table = None

    @torch.no_grad()
    def __call__(self, x):
        '''
        input is pytorch tensor of nchw
        '''
        x_fft = torch.fft.fft2(x, norm='ortho')
        x_fft_shift = torch.fft.fftshift(x_fft)
        fft_mask = self.gen_mask_circle(x_fft_shift)
        x_fft_filter = fft_mask * x_fft_shift
        x_fft_ishift = torch.fft.ifftshift(x_fft_filter)
        x_ifft = torch.fft.ifft2(x_fft_ishift, norm='ortho').abs()
        #  return x_ifft, x_fft_shift, fft_mask

        ## one channel target replace x_fft_shift
        gray = x[:, 0:1, ...] * 0.299 + x[:, 1:2, ...] * 0.587 + x[:, 2:3, ...] * 0.114
        gray_fft = torch.fft.fft2(gray, norm='ortho')
        gray_fft_shift = torch.fft.fftshift(gray_fft)
        return x_ifft, gray_fft_shift, fft_mask


    @torch.no_grad()
    def gen_mask_table(self, x):
        device = x.device
        rad = self.rad ** 2
        H, W = x.size()[-2:]
        cH, cW = H / 2., W / 2.
        coors_h = torch.arange(H, device=device)
        coors_w = torch.arange(W, device=device)
        coors = torch.stack(torch.meshgrid(coors_h, coors_w, indexing="ij"))
        coors_flatten = torch.flatten(coors, 1) # 2 x (hw)
        center = torch.tensor([cH, cW], device=device)[:, None]
        dis = (coors_flatten - center).pow(2).sum(dim=0)
        coors_center = coors_flatten[:, dis < rad]
        mask = torch.zeros((H, W), device=device)
        mask[coors_center[0], coors_center[1]] = 1.
        mask = mask.unsqueeze(0) # 1hw
        mask_table = torch.cat([mask, 1.-mask], dim=0)
        self.mask_table = mask_table.unsqueeze(1) # 21hw
        self.mask_size = (H, W)

    @torch.no_grad()
    def gen_mask_circle(self, x):
        '''
            x is nchw
        '''
        device = x.device
        batchsize = x.size(0)
        n_chan = x.size(1)
        size = tuple(x.size()[-2:])
        if self.mask_table is None or size != self.mask_size:
            self.gen_mask_table(x)
        inds = torch.randint(0, 2, (batchsize, ), device=device)
        batch_mask = self.mask_table[inds] # n1hw
        #  inds = torch.randint(0, 2, (batchsize, n_chan), device=device)
        #  batch_mask = self.mask_table.squeeze(1)[inds] # nchw
        return batch_mask

