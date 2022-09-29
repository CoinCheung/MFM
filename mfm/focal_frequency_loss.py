import torch


class FocalFrequencyLoss(torch.nn.Module):

    def __init__(self, gamma=1.):
        super().__init__()
        self.gamma = gamma


    #  def forward(self, preds, target, mask=None):
    #      '''
    #      preds is nchw real tensor
    #      target is nchw complex tensor
    #      mask is n1hw real tensor, we should use (1-mask) if we reconstruct masked portion
    #      '''
    #      preds = preds.float()
    #      p_fft = torch.fft.fft2(preds)
    #      p_fft_shift = torch.fft.fftshift(p_fft)
    #      diff = target - p_fft_shift
    #
    #      loss = diff.abs()
    #      weight = loss.clone().pow(self.gamma).detach()
    #      loss = loss * weight
    #      #  loss = diff.abs().pow(self.gamma)
    #      #  loss = (diff.real.pow(2.) + diff.imag.pow(2.)).pow(self.gamma/2)
    #      if not mask is None:
    #          mask = mask.expand_as(preds)
    #          loss = loss * (1. - mask)
    #          loss = loss.sum(dim=(1,2,3))
    #          n_pixel = (1. - mask).sum(dim=(1,2,3))
    #          loss = loss / n_pixel
    #      loss = loss.mean()
    #      return loss
    #

    #  def forward(self, preds, target, mask=None):
    #      '''
    #      preds is nchw real tensor
    #      target is nchw complex tensor
    #      mask is n1hw real tensor, we should use (1-mask) if we reconstruct masked portion
    #      '''
    #      preds = preds.float()
    #      p_fft = torch.fft.fft2(preds)
    #      p_fft_shift = torch.fft.fftshift(p_fft)
    #      diff = target - p_fft_shift
    #
    #      loss = diff.abs()
    #      weight = loss.clone().pow(self.gamma).detach()
    #      loss = loss.pow(2)
    #
    #      if not mask is None:
    #          weight = weight * (1. - mask)
    #      loss = loss * weight
    #      loss = loss.nansum() / weight.nansum()
    #
    #      return loss

    def forward(self, preds, target, mask=None):
        '''
        preds is nchw real tensor
        target is nchw complex tensor
        mask is n1hw real tensor, we should use (1-mask) if we reconstruct masked portion
        '''
        preds = preds.float()
        p_fft = torch.fft.fft2(preds)
        p_fft_shift = torch.fft.fftshift(p_fft)

        target = target.detach()
        #  replace = (target + p_fft_shift) / 2.
        #  p_fft_shift = replace * mask + p_fft_shift * (1. - mask)

        mask = (1 - mask).expand_as(preds).bool()
        p_fft_shift = p_fft_shift[mask]
        target = target[mask]

        # avg spectrum
        #  p_fft_shift = p_fft_shift.mean(dim=0, keepdim=True)
        #  target = target.mean(dim=0, keepdim=True)

        ## pow(2) first, then split and sum
        #  diff = (target - p_fft_shift).pow(2.)
        #  diff = diff.real + diff.imag

        ## split first, then pow(2) and sum
        diff = target - p_fft_shift
        diff = torch.stack([diff.real, diff.imag], dim=-1).pow(2.)
        diff = diff[..., 0] + diff[..., 1]

        #  diff = diff[(1 - mask.expand_as(diff)).bool()]

        with torch.no_grad():
            weight = diff.pow(self.gamma)
            # adjust specturm by log
            weight = torch.log(weight + 1.)
            # use batch-based statistics to compute spectrum weight
            batch_matrix = True
            if batch_matrix:
                weight = weight / weight.max()
            else:
                n, c = weight.size()[:2]
                weight = weight / weight.flatten(1).max(dim=-1, keepdim=True).values
                weight = weight.reshape(n, c, 1, 1)
            # fix bad values
            weight[weight.isnan()] = 0.
            weight = weight.clamp(min=0., max=1.)

        loss = diff * weight
        return loss.mean()
