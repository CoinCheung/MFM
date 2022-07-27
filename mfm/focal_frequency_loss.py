import torch


class FocalFrequencyLoss(torch.nn.Module):

    def __init__(self, gamma=1.):
        super().__init__()
        self.gamma = gamma


    def forward(self, preds, target, mask=None):
        '''
        preds is nchw real tensor
        target is nchw complex tensor
        mask is n1hw real tensor, we should use (1-mask) if we reconstruct masked portion
        '''
        preds = preds.float()
        p_fft = torch.fft.fft2(preds)
        p_fft_shift = torch.fft.fftshift(p_fft)
        diff = target - p_fft_shift
        loss = diff.abs().pow(self.gamma)
        if not mask is None:
            mask = mask.expand_as(preds)
            loss = loss * (1. - mask)
            loss = loss.sum(dim=(1,2,3))
            n_pixel = mask.sum(dim=(1,2,3))
            loss = loss / n_pixel
        loss = loss.mean()
        return loss

