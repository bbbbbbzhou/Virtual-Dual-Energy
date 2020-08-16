import os
import numpy as np
from math import log10
from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
from scipy.special import entr
import pdb

from networks import get_generator, get_discriminator, get_gradoperator
from networks.network_msunet import gaussian_weights_init
from models.utils import LSGANLoss
from models.utils import AverageMeter, get_scheduler, get_gan_loss, psnr, mse, get_nonlinearity
from skimage.measure import compare_ssim as ssim


class GANModel(nn.Module):
    def __init__(self, opts):
        super(GANModel, self).__init__()

        self.netG = get_generator(opts.net_G, opts)
        self.netD = get_discriminator(opts.net_D, opts)
        self.Sobelx, self.Sobely = get_gradoperator(opts)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=opts.lr, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opts.lr, betas=(opts.beta1, opts.beta2), weight_decay=opts.weight_decay)

        self.criterion_GAN = LSGANLoss().cuda(opts.gpu_ids[0])
        self.criterion_recon = nn.L1Loss().cuda(opts.gpu_ids[0])
        self.wr_recon = opts.wr_recon

        self.loss_names = ['loss_D', 'loss_G_GAN', 'loss_G_recon']
        self.opts = opts

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def initialize(self):
        self.netD.apply(gaussian_weights_init)

    def set_scheduler(self, opts, epoch=-1):
        self.scheduler_G = get_scheduler(self.optimizer_G, opts)
        self.scheduler_D = get_scheduler(self.optimizer_D, opts)

    def set_input(self, data):
        self.IH = data['IH'].to(self.device).float()
        self.IB = data['IB'].to(self.device).float()
        self.IS = data['IS'].to(self.device).float()

        if self.opts.model_type == 'model_bone':
            self.IT = self.IB
        elif self.opts.model_type == 'model_softtissue':
            self.IT = self.IS

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def forward(self, input):
        self.IT_fake = self.netG(input)
        return self.IT_fake

    def optimize(self):
        self.netD.zero_grad()
        self.IH_gradx = self.Sobelx(self.IH)
        self.IH_grady = self.Sobely(self.IH)
        self.IT_gradx = self.Sobelx(self.IT)
        self.IT_grady = self.Sobely(self.IT)

        # fake
        self.IT_fake = self.netG(self.IH)
        self.IT_fake_gradx = self.Sobelx(self.IT_fake)
        self.IT_fake_grady = self.Sobely(self.IT_fake)
        pred_fake = self.netD(torch.cat((self.IT_fake_gradx.detach(), self.IT_fake_grady.detach(), self.IT_fake.detach(),
                                         self.IH_gradx, self.IH_grady, self.IH), 1))
        loss_D_fake = self.criterion_GAN(pred_fake, target_is_real=False)

        # real
        pred_real = self.netD(torch.cat((self.IT_gradx, self.IT_grady, self.IT,
                                         self.IH_gradx, self.IH_grady, self.IH), 1))
        loss_D_real = self.criterion_GAN(pred_real, target_is_real=True)
        self.loss_D = (loss_D_fake + loss_D_real) * 0.5
        self.loss_D.backward()
        self.optimizer_D.step()

        # adv
        self.netG.zero_grad()
        self.IT_fake = self.netG(self.IH)
        pred_fake = self.netD(torch.cat((self.IT_fake_gradx, self.IT_fake_grady, self.IT_fake,
                                         self.IH_gradx, self.IH_grady, self.IH), 1))
        self.loss_G_GAN = self.criterion_GAN(pred_fake, target_is_real=True)

        self.loss_G_recon = self.criterion_recon(self.IT_fake, self.IT)
        self.loss_G = self.loss_G_GAN + self.loss_G_recon * self.wr_recon
        self.loss_G.backward()
        self.optimizer_G.step()

        # self.netD.zero_grad()
        # # fake
        # self.IT_fake = self.netG(self.IH)
        # pred_fake = self.netD(self.IT_fake.detach())
        # loss_D_fake = self.criterion_GAN(pred_fake, target_is_real=False)
        #
        # # real
        # pred_real = self.netD(self.IT)
        # loss_D_real = self.criterion_GAN(pred_real, target_is_real=True)
        # self.loss_D = (loss_D_fake + loss_D_real) * 0.5
        # self.loss_D.backward()
        # self.optimizer_D.step()
        #
        # # adv
        # self.netG.zero_grad()
        # self.IT_fake = self.netG(self.IH)
        # pred_fake = self.netD(self.IT_fake)
        # self.loss_G_GAN = self.criterion_GAN(pred_fake, target_is_real=True)
        #
        # self.loss_G_recon = self.criterion_recon(self.IT_fake, self.IT)
        # self.loss_G = self.loss_G_GAN + self.loss_G_recon * self.wr_recon
        # self.loss_G.backward()
        # self.optimizer_G.step()

    @property
    def loss_summary(self):
        return 'loss_D: {:4f}, loss_G(GAN): {:4f}, loss_G(recon): {:4f}'.format(self.loss_D.item(),
                                                                                self.loss_G_GAN.item(),
                                                                                self.loss_G_recon.item())

    def update_learning_rate(self):
        pass

    def save(self, checkpoint_dir, epoch, total_iter):
        torch.save({'netG': self.netG.module.state_dict(),
                    'optimizer_G': self.optimizer_G.state_dict(),
                    'netD': self.netD.module.state_dict(),
                    'optimizer_D': self.optimizer_D.state_dict(),
                    'epoch': epoch,
                    'total_iter': total_iter},
                   checkpoint_dir)

    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file, map_location='cuda:0')
        self.netG.module.load_state_dict(checkpoint['netG'])
        self.netD.module.load_state_dict(checkpoint['netD'])
        if train:
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])

        print('Loaded {}'.format(checkpoint_file))

        return checkpoint['epoch'], checkpoint['total_iter']

    def evaluate(self, loader):
        val_bar = tqdm(loader)
        avg_psnr = AverageMeter()
        avg_ssim = AverageMeter()
        avg_mse = AverageMeter()

        pred_images = []
        gt_images = []
        gt_inp_images = []

        for data in val_bar:
            self.set_input(data)
            self.forward(self.IH)

            psnr_ = psnr(self.IT_fake+1, self.IT+1)
            mse_ = mse(self.IT_fake+1, self.IT+1)
            ssim_ = ssim(self.IT_fake[0,0,...].cpu().numpy()+1, self.IT[0,0,...].cpu().numpy()+1)
            avg_psnr.update(psnr_)
            avg_mse.update(mse_)
            avg_ssim.update(ssim_)

            pred_images.append(self.IT_fake[0].cpu())
            gt_images.append(self.IT[0].cpu())
            gt_inp_images.append(self.IH[0].cpu())

            message = 'PSNR: {:4f} '.format(avg_psnr.avg)
            message += 'SSIM: {:4f} '.format(avg_ssim.avg)
            message += 'MSE: {:4f} '.format(avg_mse.avg)
            val_bar.set_description(desc=message)

        self.psnr = avg_psnr.avg
        self.ssim = avg_ssim.avg
        self.mse = avg_mse.avg

        self.results = {}
        self.results['pred_IT'] = torch.stack(pred_images).squeeze().numpy()
        self.results['gt_IT'] = torch.stack(gt_images).squeeze().numpy()
        self.results['gt_IH'] = torch.stack(gt_inp_images).squeeze().numpy()
