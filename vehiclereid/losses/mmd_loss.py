#!/usr/bin/env python
# encoding: utf-8
import torch
import torch.nn as nn


class MMDLoss(nn.Module):
    def __init__(self, gaussian_multi):
        """
        MMD loss.
        :param kernel: kind of mmd kernel('linear', 'poly' or 'gaussian')
        :param c: param c for poly kernel
        :param gaussian_multi: param of gamma in Gaussian kernel
        """
        super(MMDLoss, self).__init__()
        self.gaussian_multi = gaussian_multi

    def forward(self, sources, targets):
        # here inputs and targets are batches of images with shape[B,C,W,H]
        assert(sources.size(0) == targets.size(0))
        B = sources.size(0)
        x = sources.view(sources.size(0), sources.size(1) * sources.size(2) * sources.size(3))
        y = targets.view(targets.size(0), targets.size(1) * targets.size(2) * targets.size(3))
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        K = torch.exp(- self.alpha * (rx.t() + rx - 2 * xx))
        L = torch.exp(- self.alpha * (ry.t() + ry - 2 * yy))
        P = torch.exp(- self.alpha * (rx.t() + ry - 2 * zz))

        beta = (1. / (B * (B - 1)))
        gamma = (2. / (B * B))

        return beta * (torch.sum(K) + torch.sum(L)) - gamma * torch.sum(P)

