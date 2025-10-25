import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import numpy as np
import cv2
from models.archs.CFblock import *
from  models.archs import build_model
import time
###############################
class low_light_mamba(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True):
        super(low_light_mamba, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        if self.HR_in:
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)

        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)

        self.upconv1 = nn.Conv2d(nf*2, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf*2, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64*2, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.recon_trunk_light = arch_util.make_layer(ResidualBlock_noBN_f, 6)
        self.vmamba = build_model()
        self.select_gate = Generate_gate()
        self.select_gate2 = Generate_gate()
        self.select_gate3 =  Generate_gate()
        self.fusegrad2 = nn.Conv2d(nf+12, nf, 3, 1, 1, bias=True)
        self.conv_nf = nn.Conv2d(3, nf, 3, 1, 1, bias=True)

    def forward(self, x,nf,nfg):
        x_center = x
        center = nf[:,:,1:-1,1:-1]
        left = nf[:,:,0:-2,1:-1]
        right = nf[:, :, 2:, 1:-1]
        top = nf[:, :, 1:-1, 0:-2]
        bottom = nf[:, :, 1:-1, 2:]
        nfea= self.conv_nf(nf)
        L1_fea_1 = self.lrelu(self.conv_first_1(x_center))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))
        gate= self.select_gate(L1_fea_1)
        L1_fea_1 = gate*nfea + (1-gate)*L1_fea_1
        height = L1_fea_2.shape[2]
        width = L1_fea_2.shape[3]
        top_g = F.interpolate(center - top, size=(height, width))
        bottom_g = F.interpolate(center - bottom, size=(height, width))
        left_g = F.interpolate(center - left, size=(height, width))
        right_g = F.interpolate(center - right, size=(height, width))
        L2_fea_grad = torch.cat([top_g , bottom_g , left_g , right_g],dim=1)
        nfL1_fea_2 = self.fusegrad2(torch.cat([L1_fea_2,L2_fea_grad], dim=1))
        gate2 = self.select_gate2(nfL1_fea_2)
        L1_fea_2 = gate2*nfL1_fea_2 + (1-gate2)*L1_fea_2
        height = L1_fea_3.shape[2]
        width = L1_fea_3.shape[3]
        nfg = F.interpolate(nfg, size=(height, width))
        nfgavg = (nfg[:,0:1,:,:]+ nfg[:,1:2,:,:]+nfg[:,2:3,:,:])/3

        fea = self.feature_extraction(L1_fea_3)
        nffea = L1_fea_3 + nfgavg
        gate3 = self.select_gate3(nffea)
        fea = gate3 * nffea + (1 - gate3) * fea
        fea_light = self.recon_trunk_light(fea)
        fea_long=self.vmamba(fea)
        fea = fea_long+fea_light
        out_noise = self.recon_trunk(fea)
        out_noise = torch.cat([out_noise, L1_fea_3], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
        out_noise = self.lrelu(self.HRconv(out_noise))
        out_noise = self.conv_last(out_noise)
        out_noise = out_noise + x_center
        return out_noise
