

import random
import torch


class MaskedFFT:

    def __init__(self, rad=16):
        self.rad = rad

    def __call__(self, x):
        '''
        input is pytorch tensor of chw
        '''
        x_fft = torch.fft.fft2(x)
        x_fft_shift = torch.fft.fftshift(x_fft)
        fft_mask = self.gen_mask_circle(x_fft_shift)
        if random.random() > 0.5: fft_mask = (1. - fft_mask)
        x_fft_filter = fft_mask * x_fft_shift
        x_fft_ishift = torch.fft.ifftshift(x_fft_filter)
        x_ifft = torch.fft.ifft2(x_fft_ishift).abs()
        return x_ifft, x_fft_shift, fft_mask

    ## TODO: 1.see if we can generate mask only once and reuse it
    ##  2. see if we can move this to main code and use gpu
    def gen_mask_circle(self, x):
        '''
            x is chw
        '''
        device = x.device
        rad = self.rad ** 2
        H, W = x.size()[-2:]
        cH, cW = H / 2., W / 2.
        coors_h = torch.arange(H, device=device)
        coors_w = torch.arange(W, device=device)
        coors = torch.stack(torch.meshgrid(coors_h, coors_w, indexing="ij"))
        coors_flatten = torch.flatten(coors, 1) # 2 x (hw)
        center = torch.tensor([cH, cW])[:, None]
        dis = (coors_flatten - center).pow(2).sum(dim=0)
        coors_center = coors_flatten[:, dis < rad]
        mask = torch.zeros((H, W), device=device)
        mask[coors_center[0], coors_center[1]] = 1
        mask = mask.unsqueeze(0) # 1hw
        return mask
