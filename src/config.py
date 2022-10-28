import json
import os
import random
import sys
import yaml

import torch
import torch.nn as nn

import ops
import misc

class Configurations(object):
    def __init__(self, cfg_file):
        self.cfg_file = cfg_file
        self.load_base_cfgs()
        self._overwrite_cfgs(self.cfg_file)
        self.define_modules()


    def load_base_cfgs(self):
        # -----------------------------------------------------------------------------
        # Data settings
        # -----------------------------------------------------------------------------
        self.DATA = misc.make_empty_object()

        # dataset name \in ["CIFAR10", "CIFAR100", "Tiny_ImageNet", "CUB200", "ImageNet", "MY_DATASET"]
        self.DATA.name = "CIFAR10"
        # image size for training
        self.DATA.img_size = 32
        # number of classes in training dataset, if there is no explicit class label, DATA.num_classes = 1
        self.DATA.num_classes = 10
        # number of image channels in dataset. //image_shape[0]
        self.DATA.img_channels = 3

        # -----------------------------------------------------------------------------
        # Model settings
        # -----------------------------------------------------------------------------
        self.MODEL = misc.make_empty_object()

        # type of backbone architectures of the generator and discriminator \in ["deep_conv", "resnet", "big_resnet", "deep_big_resnet", "stylegan2"]
        self.MODEL.backbone = "resnet"
        # conditioning method of the generator \in ["W/O", "cBN", "cAdaIN"]
        self.MODEL.g_cond_mtd = "W/O"
        # conditioning method of the discriminator \in ["W/O", "AC", "PD", "MH", "MD", "2C","D2DCE", "SPD"]
        self.MODEL.d_cond_mtd = "W/O"
        # type of auxiliary classifier \in ["W/O", "TAC", "ADC"]
        self.MODEL.aux_cls_type = "W/O"
        # whether to normalize feature maps from the discriminator or not
        self.MODEL.normalize_d_embed = False
        # dimension of feature maps from the discriminator
        # only appliable when MODEL.d_cond_mtd \in ["2C, D2DCE"]
        self.MODEL.d_embed_dim = "N/A"
        # whether to apply spectral normalization on the generator
        self.MODEL.apply_g_sn = False
        # whether to apply spectral normalization on the discriminator
        self.MODEL.apply_d_sn = False
        # type of activation function in the generator \in ["ReLU", "Leaky_ReLU", "ELU", "GELU"]
        self.MODEL.g_act_fn = "ReLU"
        # type of activation function in the discriminator \in ["ReLU", "Leaky_ReLU", "ELU", "GELU"]
        self.MODEL.d_act_fn = "ReLU"
        # whether to apply self-attention proposed by zhang et al. (SAGAN)
        self.MODEL.apply_attn = False
        # locations of the self-attention layer in the generator (should be list type)
        self.MODEL.attn_g_loc = ["N/A"]
        # locations of the self-attention layer in the discriminator (should be list type)
        self.MODEL.attn_d_loc = ["N/A"]
        # prior distribution for noise sampling \in ["gaussian", "uniform"]
        self.MODEL.z_prior = "gaussian"
        # dimension of noise vectors
        self.MODEL.z_dim = 128
        # dimension of intermediate latent (W) dimensionality used only for StyleGAN
        self.MODEL.w_dim = "N/A"
        # dimension of a shared latent embedding
        self.MODEL.g_shared_dim = "N/A"
        # base channel for the resnet style generator architecture
        self.MODEL.g_conv_dim = 64
        # base channel for the resnet style discriminator architecture
        self.MODEL.d_conv_dim = 64
        # generator's depth for deep_big_resnet
        self.MODEL.g_depth = "N/A"
        # discriminator's depth for deep_big_resnet
        self.MODEL.d_depth = "N/A"
        # whether to apply moving average update for the generator
        self.MODEL.apply_g_ema = False
        # decay rate for the ema generator
        self.MODEL.g_ema_decay = "N/A"
        # starting step for g_ema update
        self.MODEL.g_ema_start = "N/A"
        # weight initialization method for the generator \in ["ortho", "N02", "glorot", "xavier"]
        self.MODEL.g_init = "ortho"
        # weight initialization method for the discriminator \in ["ortho", "N02", "glorot", "xavier"]
        self.MODEL.d_init = "ortho"

        # -----------------------------------------------------------------------------
        # loss settings
        # -----------------------------------------------------------------------------
        self.LOSS = misc.make_empty_object()

        # type of adversarial loss \in ["vanilla", "least_squere", "wasserstein", "hinge", "MH"]
        self.LOSS.adv_loss = "vanilla"
        # balancing hyperparameter for conditional image generation
        self.LOSS.cond_lambda = "N/A"
        # klcp lambda
        self.LOSS.klcp_lambda = "N/A"
        # strength of conditioning loss induced by twin auxiliary classifier for generator training
        self.LOSS.tac_gen_lambda = "N/A"
        # strength of conditioning loss induced by twin auxiliary classifier for discriminator training
        self.LOSS.tac_dis_lambda = "N/A"
        # strength of multi-hinge loss (MH) for the generator training
        self.LOSS.mh_lambda = "N/A"
        # whether to apply feature matching regularization
        self.LOSS.apply_fm = False
        # strength of feature matching regularization
        self.LOSS.fm_lambda = "N/A"
        # whether to apply r1 regularization used in multiple-discriminator (FUNIT)
        self.LOSS.apply_r1_reg = False
        # strength of r1 regularization (it does not apply to r1_reg in StyleGAN2
        self.LOSS.r1_lambda = "N/A"
        # positive margin for D2DCE
        self.LOSS.m_p = "N/A"
        # temperature scalar for [2C, D2DCE]
        self.LOSS.temperature = "N/A"
        # whether to apply weight clipping regularization to let the discriminator satisfy Lipschitzness
        self.LOSS.apply_wc = False
        # clipping bound for weight clippling regularization
        self.LOSS.wc_bound = "N/A"
        # whether to apply gradient penalty regularization
        self.LOSS.apply_gp = False
        # strength of the gradient penalty regularization
        self.LOSS.gp_lambda = "N/A"
        # whether to apply deep regret analysis regularization
        self.LOSS.apply_dra = False
        # strength of the deep regret analysis regularization
        self.LOSS.dra_labmda = "N/A"
        # whther to apply max gradient penalty to let the discriminator satisfy Lipschitzness
        self.LOSS.apply_maxgp = False
        # strength of the maxgp regularization
        self.LOSS.maxgp_lambda = "N/A"
        # whether to apply consistency regularization
        self.LOSS.apply_cr = False
        # strength of the consistency regularization
        self.LOSS.cr_lambda = "N/A"
        # whether to apply balanced consistency regularization
        self.LOSS.apply_bcr = False
        # attraction strength between logits of real and augmented real samples
        self.LOSS.real_lambda = "N/A"
        # attraction strength between logits of fake and augmented fake samples
        self.LOSS.fake_lambda = "N/A"
        # whether to apply latent consistency regularization
        self.LOSS.apply_zcr = False
        # radius of ball to generate an fake image G(z + radius)
        self.LOSS.radius = "N/A"
        # repulsion stength between fake images (G(z), G(z + radius))
        self.LOSS.g_lambda = "N/A"
        # attaction stength between logits of fake images (G(z), G(z + radius))
        self.LOSS.d_lambda = "N/A"
        # whether to apply latent optimization for stable training
        self.LOSS.apply_lo = False
        # latent step size for latent optimization
        self.LOSS.lo_alpha = "N/A"
        # damping factor for calculating Fisher Information matrix
        self.LOSS.lo_beta = "N/A"
        # portion of z for latent optimization (c)
        self.LOSS.lo_rate = "N/A"
        # strength of latent optimization (w_{r})
        self.LOSS.lo_lambda = "N/A"
        # number of latent optimization iterations for a single sample during training
        self.LOSS.lo_steps4train = "N/A"
        # number of latent optimization iterations for a single sample during evaluation
        self.LOSS.lo_steps4eval = "N/A"
        # whether to apply topk training for the generator update
        self.LOSS.apply_topk = False
        # hyperparameter for batch_size decay rate for topk training \in [0,1]
        self.LOSS.topk_gamma = "N/A"
        # hyperparameter for the supremum of the number of topk samples \in [0,1],
        # sup_batch_size = int(topk_nu*batch_size)
        self.LOSS.topk_nu = "N/A"

        # -----------------------------------------------------------------------------
        # optimizer settings
        # -----------------------------------------------------------------------------
        self.OPTIMIZATION = misc.make_empty_object()

        # type of the optimizer for GAN training \in ["SGD", RMSprop, "Adam"]
        self.OPTIMIZATION.type_ = "Adam"
        # number of batch size for GAN training,
        # typically {CIFAR10: 64, CIFAR100: 64, Tiny_ImageNet: 1024, "CUB200": 256, ImageNet: 512(batch_size) * 4(accm_step)"}
        self.OPTIMIZATION.batch_size = 64
        # acuumulation steps for large batch training (batch_size = batch_size*accm_step)
        self.OPTIMIZATION.acml_steps = 1
        # learning rate for generator update
        self.OPTIMIZATION.g_lr = 0.0002
        # learning rate for discriminator update
        self.OPTIMIZATION.d_lr = 0.0002
        # weight decay strength for the generator update
        self.OPTIMIZATION.g_weight_decay = 0.0
        # weight decay strength for the discriminator update
        self.OPTIMIZATION.d_weight_decay = 0.0
        # momentum value for SGD and RMSprop optimizers
        self.OPTIMIZATION.momentum = "N/A"
        # nesterov value for SGD optimizer
        self.OPTIMIZATION.nesterov = "N/A"
        # alpha value for RMSprop optimizer
        self.OPTIMIZATION.alpha = "N/A"
        # beta values for Adam optimizer
        self.OPTIMIZATION.beta1 = 0.5
        self.OPTIMIZATION.beta2 = 0.999
        # the number of generator updates per step
        self.OPTIMIZATION.g_updates_per_step = 1
        # the number of discriminator updates per step
        self.OPTIMIZATION.d_updates_per_step = 5
        # the total number of steps for GAN training
        self.OPTIMIZATION.total_steps = 100000

        # -----------------------------------------------------------------------------
        # preprocessing settings
        # -----------------------------------------------------------------------------
        self.PRE = misc.make_empty_object()

        # whether to apply random flip preprocessing before training
        self.PRE.apply_rflip = True

        # -----------------------------------------------------------------------------
        # differentiable augmentation settings
        # -----------------------------------------------------------------------------
        self.AUG = misc.make_empty_object()

        # whether to apply differentiable augmentations for limited data training
        self.AUG.apply_diffaug = False
        # whether to apply adaptive discriminator augmentation (ADA)
        self.AUG.apply_ada = False
        # type of differentiable augmentation for cr, bcr, or limited data training
        # \in ["W/O", "cr", "bcr", "diffaug", "simclr_basic", "simclr_hq", "simclr_hq_cutout", "byol"
        # \ "blit", "geom", "color", "filter", "noise", "cutout", "bg", "bgc", "bgcf", "bgcfn", "bgcfnc"]
        # "blit", "geon", ... "bgcfnc" augmentations details are available at ...
        # for ada default aug_type is bgc, ada_target is 0.6, ada_kimg is 500.
        # cr (bcr, diffaugment, ada, simclr, byol) indicates differentiable augmenations used in the original paper
        self.AUG.cr_aug_type = "W/O"
        self.AUG.bcr_aug_type = "W/O"
        self.AUG.diffaug_type = "W/O"
        self.AUG.ada_aug_type = "W/O"
        # initial value of augmentation probability.
        self.AUG.ada_initial_augment_p = "N/A"
        # target probability for adaptive differentiable augmentations, None = fixed p (keep ada_initial_augment_p)
        self.AUG.ada_target = "N/A"
        # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
        self.AUG.ada_kimg = "N/A"
        # how often to perform ada adjustment
        self.AUG.ada_interval = "N/A"

        # -----------------------------------------------------------------------------
        # StyleGAN_v2 settings regarding regularization and style mixing
        # selected configurations by official implementation is given below.
        # 'paper256':  dict(gpus=8,  total_steps=390,625,   batch_size=64, d_epilogue_mbstd_group_size=8,  g/d_lr=0.0025,
        #                   r1_lambda=1,    g_ema_kimg=20,  g_ema_rampup=None, mapping_network=8),
        # 'paper512':  dict(gpus=8,  total_steps=390,625,   batch_size=64, d_epilogue_mbstd_group_size=8,  g/d_lr=0.0025,
        #                   r1_lambda=0.5,  g_ema_kimg=20,  g_ema_rampup=None, mapping_network=8),
        # 'paper1024': dict(gpus=8,  total_steps=781,250,   batch_size=32, d_epilogue_mbstd_group_size=4,  g/d_lr=0.002,
        #                   r1_lambda=2,    g_ema_kimg=10,  g_ema_rampup=None, mapping_network=8),
        # 'cifar':     dict(gpus=2,  total_steps=1,562,500, batch_size=64, d_epilogue_mbstd_group_size=32, g/d_lr=0.0025,
        #                   r1_lambda=0.01, g_ema_kimg=500, g_ema_rampup=0.05, mapping_network=2),
        # -----------------------------------------------------------------------------
        self.STYLEGAN2 = misc.make_empty_object()

        # conditioning types that utilize embedding proxies for conditional stylegan2
        self.STYLEGAN2.cond_type = ["PD", "SPD", "2C", "D2DCE"]
        # lazy regularization interval for generator, default 4
        self.STYLEGAN2.g_reg_interval = "N/A"
        # lazy regularization interval for discriminator, default 16
        self.STYLEGAN2.d_reg_interval = "N/A"
        # number of layers for the mapping network, default 8 except for cifar (2)
        self.STYLEGAN2.mapping_network = "N/A"
        # style_mixing_p in stylegan generator, default 0.9 except for cifar (0)
        self.STYLEGAN2.style_mixing_p = "N/A"
        # half-life of the exponential moving average (EMA) of generator weights default 500
        self.STYLEGAN2.g_ema_kimg = "N/A"
        # EMA ramp-up coefficient, defalt "N/A" except for cifar 0.05
        self.STYLEGAN2.g_ema_rampup = "N/A"
        # whether to apply path length regularization, default is True except cifar
        self.STYLEGAN2.apply_pl_reg = False
        # pl regularization strength, default 2
        self.STYLEGAN2.pl_weight = "N/A"
        # discriminator architecture for STYLEGAN2. 'resnet' except for cifar10 ('orig')
        self.STYLEGAN2.d_architecture = "N/A"
        # group size for the minibatch standard deviation layer, None = entire minibatch.
        self.STYLEGAN2.d_epilogue_mbstd_group_size = "N/A"

        # -----------------------------------------------------------------------------
        # run settings
        # -----------------------------------------------------------------------------
        self.RUN = misc.make_empty_object()

        # -----------------------------------------------------------------------------
        # run settings
        # -----------------------------------------------------------------------------
        self.MISC = misc.make_empty_object()

        self.MISC.no_proc_data = ["CIFAR10", "CIFAR100", "Tiny_ImageNet"]
        self.MISC.base_folders = ["checkpoints", "figures", "logs", "moments", "samples", "values"]
        self.MISC.classifier_based_GAN = ["AC", "2C", "D2DCE"]
        self.MISC.cas_setting = {
            "CIFAR10": {
                "batch_size": 128,
                "epochs": 90,
                "depth": 32,
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "print_freq": 1,
                "bottleneck": True
            },
            "Tiny_ImageNet": {
                "batch_size": 128,
                "epochs": 90,
                "depth": 34,
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "print_freq": 1,
                "bottleneck": True
            },
            "ImageNet": {
                "batch_size": 128,
                "epochs": 90,
                "depth": 34,
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "print_freq": 1,
                "bottleneck": True
            },
        }

        # -----------------------------------------------------------------------------
        # Module settings
        # -----------------------------------------------------------------------------
        self.MODULES = misc.make_empty_object()

        self.super_cfgs = {
            "DATA": self.DATA,
            "MODEL": self.MODEL,
            "LOSS": self.LOSS,
            "OPTIMIZATION": self.OPTIMIZATION,
            "PRE": self.PRE,
            "AUG": self.AUG,
            "RUN": self.RUN,
            "STYLEGAN2": self.STYLEGAN2
        }

    def update_cfgs(self, cfgs, super="RUN"):
        for attr, value in cfgs.items():
            setattr(self.super_cfgs[super], attr, value)

    def _overwrite_cfgs(self, cfg_file):
        with open(cfg_file, 'r') as f:
            yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
            for super_cfg_name, attr_value in yaml_cfg.items():
                for attr, value in attr_value.items():
                    if hasattr(self.super_cfgs[super_cfg_name], attr):
                        setattr(self.super_cfgs[super_cfg_name], attr, value)
                    else:
                        raise AttributeError("There does not exist '{cls}.{attr}' attribute in the config.py.". \
                                             format(cls=super_cfg_name, attr=attr))

    def define_modules(self):
        if self.MODEL.apply_g_sn:
            self.MODULES.g_conv2d = ops.snconv2d
            self.MODULES.g_deconv2d = ops.sndeconv2d
            self.MODULES.g_linear = ops.snlinear
            self.MODULES.g_embedding = ops.sn_embedding
        else:
            self.MODULES.g_conv2d = ops.conv2d
            self.MODULES.g_deconv2d = ops.deconv2d
            self.MODULES.g_linear = ops.linear
            self.MODULES.g_embedding = ops.embedding

        if self.MODEL.apply_d_sn:
            self.MODULES.d_conv2d = ops.snconv2d
            self.MODULES.d_deconv2d = ops.sndeconv2d
            self.MODULES.d_linear = ops.snlinear
            self.MODULES.d_embedding = ops.sn_embedding
        else:
            self.MODULES.d_conv2d = ops.conv2d
            self.MODULES.d_deconv2d = ops.deconv2d
            self.MODULES.d_linear = ops.linear
            self.MODULES.d_embedding = ops.embedding

        if self.MODEL.g_cond_mtd == "cBN" and self.MODEL.backbone in ["big_resnet", "deep_big_resnet"]:
            self.MODULES.g_bn = ops.BigGANConditionalBatchNorm2d
        elif self.MODEL.g_cond_mtd == "cBN":
            self.MODULES.g_bn = ops.ConditionalBatchNorm2d
        elif self.MODEL.g_cond_mtd == "W/O":
            self.MODULES.g_bn = ops.batchnorm_2d
        elif self.MODEL.g_cond_mtd == "cAdaIN":
            pass
        else:
            raise NotImplementedError

        if not self.MODEL.apply_d_sn:
            self.MODULES.d_bn = ops.batchnorm_2d

        if self.MODEL.g_act_fn == "ReLU":
            self.MODULES.g_act_fn = nn.ReLU(inplace=True)
        elif self.MODEL.g_act_fn == "Leaky_ReLU":
            self.MODULES.g_act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif self.MODEL.g_act_fn == "ELU":
            self.MODULES.g_act_fn = nn.ELU(alpha=1.0, inplace=True)
        elif self.MODEL.g_act_fn == "GELU":
            self.MODULES.g_act_fn = nn.GELU()
        elif self.MODEL.g_act_fn == "Auto":
            pass
        else:
            raise NotImplementedError

        if self.MODEL.d_act_fn == "ReLU":
            self.MODULES.d_act_fn = nn.ReLU(inplace=True)
        elif self.MODEL.d_act_fn == "Leaky_ReLU":
            self.MODULES.d_act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif self.MODEL.d_act_fn == "ELU":
            self.MODULES.d_act_fn = nn.ELU(alpha=1.0, inplace=True)
        elif self.MODEL.d_act_fn == "GELU":
            self.MODULES.d_act_fn = nn.GELU()
        elif self.MODEL.g_act_fn == "Auto":
            pass
        else:
            raise NotImplementedError
        return self.MODULES

