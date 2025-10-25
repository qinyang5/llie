import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, CharbonnierLoss2,cosloss,freqloss
import models.pytorch_ssim as pytorch_ssim
import time
import numpy as np
logger = logging.getLogger('base')


class VideoBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoBaseModel, self).__init__(opt)

        self.fake_H = None
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction='sum').to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            elif loss_type == 'cb2':
                self.cri_pix = CharbonnierLoss2().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            # self.l_pix_w = train_opt['pixel_weight']
            self.cosloss = cosloss().to(self.device)
            self.maeloss = nn.L1Loss(reduction='sum').to(self.device)
            self.ssim_loss = pytorch_ssim.SSIM(window_size=11).to(self.device)
            self.lcdweiht = train_opt['pixel_weight']
            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['ft_tsa_only']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        self.nf = data['nf'].to(self.device)
        self.nfg = data['nfg'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()

        self.fake_H = self.netG(self.var_L,self.nf,self.nfg)


        preandgtratio = (self.fake_H[:,1,:,:]+1e-6)/(self.real_H[:,1,:,:]+1e-6)
        all_one= torch.ones_like(preandgtratio)
        l_ratio= self.cosloss(preandgtratio,all_one)
        fake12 = self.fake_H[:,0,:,:]-self.fake_H[:,1,:,:]
        fake13 = self.fake_H[:, 0, :, :] - self.fake_H[:, 2, :, :]
        fake23 = self.fake_H[:, 1, :, :] - self.fake_H[:, 2, :, :]
        real12 = self.real_H[:,0,:,:]-self.real_H[:,1,:,:]
        real13 = self.real_H[:, 0, :, :] - self.real_H[:, 2, :, :]
        real23 = self.real_H[:, 1, :, :] - self.real_H[:, 2, :, :]
        l_feainter1 =self.maeloss(fake12,real12)
        l_feainter2 = self.maeloss(fake13, real13)
        l_feainter3 = self.maeloss(fake23, real23)



        l_fea = l_feainter1+l_feainter2 + l_feainter3

        l_pix =  self.cri_pix(self.fake_H, self.real_H)

        l_ssim= 0.1*self.ssim_loss(self.fake_H,self.real_H) #0.1
        l_cd =self.lcdweiht*(l_fea+ l_ratio)
        # l_final = l_pix+l_ssim+l_cd
        l_final = l_pix + l_ssim + l_cd
        l_final.backward()
        self.optimizer_G.step()
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['l_feainter1'] = l_feainter1.item()
        self.log_dict['l_feainter2'] = l_feainter2.item()
        self.log_dict['l_feainter3'] = l_feainter3.item()
        self.log_dict['l_fea'] = l_fea.item()
        self.log_dict['l_cd'] = l_cd.item()
        self.log_dict['l_ssim'] = l_ssim.item()
        self.log_dict['l_ratio'] = l_ratio.item()
        self.log_dict['l_final'] = l_final.item()

    def self_ensemble(self, x, nf,nfg, model):
        def forward_transformed(x, nf,nfg, hflip,rotate, model):
            if hflip:
                x = torch.flip(x, (-2,))
                nf = torch.flip(nf, (-2,))
                nfg = torch.flip(nfg, (-2,))
            if rotate:
                x = torch.rot90(x, dims=(-2, -1), k=2)
            x = model(x, nf,nfg)
            if rotate:
                x = torch.rot90(x, dims=(-2, -1), k=2)
            if hflip:
                x = torch.flip(x, (-2,))
            return x

        t = []
        for hflip in [False,True]:
                    t.append(forward_transformed(x, nf,nfg, hflip,False, model))
        t = torch.stack(t)
        return torch.mean(t, dim=0)
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H=self.self_ensemble(self.var_L,self.nf,self.nfg, self.netG)
            # self.fake_H = self.netG(self.var_L, self.nf, self.nfg)
        self.netG.train()
    def testunsupervise(self):
        self.netG.eval()

        self.var_L, original_size1, padding_params1 = self.pad_to_multiples_of_32(self.var_L)
        self.nf, original_size2, padding_params2 = self.pad_to_multiples_of_32(self.nf)
        self.nfg, original_size3, padding_params3= self.pad_to_multiples_of_32(self.nfg)

        with torch.no_grad():
            # self.fake_H = self.netG(self.var_L, self.nf,self.nfg)
            self.fake_H = self.self_ensemble(self.var_L, self.nf, self.nfg, self.netG)
            self.fake_H  = self.unpad_from_multiples_of_32(self.fake_H, original_size1, padding_params1)
        self.netG.train()


    def pad_to_multiples_of_32(self,tensor):
        """
        将图像填充到32的倍数尺寸
        参数:
            tensor: 输入图像张量 (B, C, H, W)
        返回:
            padded_tensor: 填充后的张量
            (original_heights, original_widths): 原始高度和宽度列表
            padding_params: 填充参数列表 [(top, bottom, left, right), ...]
        """
        B, C, H, W = tensor.shape

        # 计算需要填充的像素数
        # pad_h = (64 - H % 64) % 64
        # pad_w = (64 - W % 64) % 64
        # pad_h = (256 - H % 256) % 256
        # pad_w = (256 - W % 256) % 256
        pad_h = (16- H % 16) % 16
        pad_w = (16 - W % 16) % 16

        # 均匀分配填充到两侧
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # 进行填充 (使用反射模式减少边界伪影)
        padded_tensor = F.pad(tensor,
                              (pad_left, pad_right, pad_top, pad_bottom),
                              mode='reflect')

        # 保存原始尺寸和填充参数
        original_heights = [H] * B
        original_widths = [W] * B
        padding_params = [(pad_top, pad_bottom, pad_left, pad_right)] * B

        return padded_tensor, (original_heights, original_widths), padding_params

    def unpad_from_multiples_of_32(self,padded_tensor, original_sizes, padding_params):
        """
        从填充后的张量恢复原始尺寸
        参数:
            padded_tensor: 填充后的张量 (B, C, H, W)
            original_sizes: (original_heights, original_widths)
            padding_params: [(top, bottom, left, right), ...]
        返回:
            恢复后的张量 (B, C, original_H, original_W)
        """
        original_heights, original_widths = original_sizes
        unpadded_tensors = []

        for i in range(padded_tensor.shape[0]):
            h = original_heights[i]
            w = original_widths[i]
            pad_top, pad_bottom, pad_left, pad_right = padding_params[i]

            # 计算裁剪区域
            start_h = pad_top
            end_h = padded_tensor.shape[2] - pad_bottom if pad_bottom > 0 else None
            start_w = pad_left
            end_w = padded_tensor.shape[3] - pad_right if pad_right > 0 else None

            # 裁剪单张图像
            unpadded = padded_tensor[i, :, start_h:end_h, start_w:end_w]
            unpadded_tensors.append(unpadded.unsqueeze(0))

        return torch.cat(unpadded_tensors, dim=0)

    def testR(self):
        self.netG.eval()
        m = nn.ReflectionPad2d(padding=(3,0,3,0))
        revm = nn.ReflectionPad2d(padding=(-3, 0, -3, 0))
        self.var_L=m(self.var_L)
        self.nf = m(self.nf)
        self.nfg = m(self.nfg)
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L, self.nf,self.nfg)
            # self.fake_H = self.self_ensemble(self.var_L, self.nf, self.nfg, self.netG)
            self.var_L = revm(self.var_L)
            self.nfg = revm(self.nfg)
            self.nf = revm(self.nf)
            self.fake_H = revm(self.fake_H)
        self.netG.train()
    def test4(self):
        self.netG.eval()
        self.fake_H = None
        with torch.no_grad():
            B, C, H, W = self.var_L.size()

            dark = self.var_L
            dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
            if not (len(self.nf.shape) == 4):
                self.nf = self.nf.unsqueeze(dim=0)
            light = self.nf
            light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
            noise = torch.abs(dark - light)
            mask = torch.div(light, noise + 0.0001)

            batch_size = mask.shape[0]
            height = mask.shape[2]
            width = mask.shape[3]
            mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
            mask_max = mask_max.view(batch_size, 1, 1, 1)
            mask_max = mask_max.repeat(1, 1, height, width)
            mask = mask * 1.0 / (mask_max + 0.0001)

            mask = torch.clamp(mask, min=0, max=1.0)
            mask = mask.float()

            del light
            del dark
            del noise
            torch.cuda.empty_cache()

            var_L = self.var_L.clone().view(B, C, H, W)
            H_new = 400
            W_new = 608
            var_L = F.interpolate(var_L, size=[H_new, W_new], mode='bilinear')
            mask = F.interpolate(mask, size=[H_new, W_new], mode='bilinear')
            var_L = var_L.view(B, C, H_new, W_new)
            self.fake_H = self.netG(var_L, mask)
            self.fake_H = F.interpolate(self.fake_H, size=[H, W], mode='bilinear')

            del var_L
            del mask
            torch.cuda.empty_cache()

        self.netG.train()


    def test5(self):
        self.netG.eval()
        self.fake_H = None
        with torch.no_grad():
            B, C, H, W = self.var_L.size()

            dark = self.var_L
            dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
            if not (len(self.nf.shape) == 4):
                self.nf = self.nf.unsqueeze(dim=0)
            light = self.nf
            light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
            noise = torch.abs(dark - light)
            mask = torch.div(light, noise + 0.0001)

            batch_size = mask.shape[0]
            height = mask.shape[2]
            width = mask.shape[3]
            mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
            mask_max = mask_max.view(batch_size, 1, 1, 1)
            mask_max = mask_max.repeat(1, 1, height, width)
            mask = mask * 1.0 / (mask_max + 0.0001)

            mask = torch.clamp(mask, min=0, max=1.0)
            mask = mask.float()

            del light
            del dark
            del noise
            torch.cuda.empty_cache()

            var_L = self.var_L.clone().view(B, C, H, W)
            H_new = 384
            W_new = 384
            var_L = F.interpolate(var_L, size=[H_new, W_new], mode='bilinear')
            mask = F.interpolate(mask, size=[H_new, W_new], mode='bilinear')
            var_L = var_L.view(B, C, H_new, W_new)
            self.fake_H = self.netG(var_L, mask)
            self.fake_H = F.interpolate(self.fake_H, size=[H, W], mode='bilinear')

            del var_L
            del mask
            torch.cuda.empty_cache()

        self.netG.train()


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        del self.real_H
        del self.var_L
        del self.fake_H
        torch.cuda.empty_cache()
        return out_dict


    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)