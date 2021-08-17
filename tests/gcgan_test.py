import numpy as np
import torch
import torch.nn as nn
import os
from collections import OrderedDict
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.nn import init
import itertools
import random
import functools


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) + 1
    image_numpy = image_numpy/2.0
    image_numpy = image_numpy*255.0
    return image_numpy.astype(imtype)


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return Variable(images)
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images


class BaseModel:
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'poly':
        def lambda_poly(epoch):
            lr_l = (1.0 - min(float(opt.iter_num) / float(opt.max_iter_num), 1.0)) ** 0.9
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_poly)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(PixelDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.net, input, self.gpu_ids)
        else:
            return self.net(input)


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    init_weights(netD, init_type=init_type)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class GcGANShareModel(BaseModel):
    def name(self):
        return 'GcGANShareModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize

        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        self.netG_AB = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_B = define_D(opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_gc_B = define_D(opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_AB, 'G_AB', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_B, 'D_B', which_epoch)
                self.load_network(self.netD_gc_B, 'D_gc_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.fake_gc_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionGc = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_AB.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(itertools.chain(self.netD_B.parameters(), self.netD_gc_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        print_network(self.netG_AB)
        if self.isTrain:
            print_network(self.netD_B)
            print_network(self.netD_gc_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def backward_D_basic(self, netD, real, fake, netD_gc, real_gc, fake_gc):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        # Real_gc
        pred_real_gc = netD_gc(real_gc)
        loss_D_gc_real = self.criterionGAN(pred_real_gc, True)
        # Fake_gc
        pred_fake_gc = netD_gc(fake_gc.detach())
        loss_D_gc_fake = self.criterionGAN(pred_fake_gc, False)
        # Combined loss
        loss_D += (loss_D_gc_real + loss_D_gc_fake) * 0.5

        # backward
        loss_D.backward()
        return loss_D

    def get_image_paths(self):
        return self.image_paths

    def rot90(self, tensor, direction):
        tensor = tensor.transpose(2, 3)
        size = self.opt.fineSize
        inv_idx = torch.arange(size - 1, -1, -1).long().cuda()
        if direction == 0:
            tensor = torch.index_select(tensor, 3, inv_idx)
        else:
            tensor = torch.index_select(tensor, 2, inv_idx)
        return tensor

    def forward(self):
        input_A = self.input_A.clone()
        input_B = self.input_B.clone()

        self.real_A = self.input_A
        self.real_B = self.input_B

        size = self.opt.fineSize

        if self.opt.geometry == 'rot':
            self.real_gc_A = self.rot90(input_A, 0)
            self.real_gc_B = self.rot90(input_B, 0)
        elif self.opt.geometry == 'vf':
            inv_idx = torch.arange(size - 1, -1, -1).long().cuda()
            self.real_gc_A = torch.index_select(input_A, 2, inv_idx)
            self.real_gc_B = torch.index_select(input_B, 2, inv_idx)
        else:
            raise ValueError("Geometry transformation function [%s] not recognized." % self.opt.geometry)

    def get_gc_rot_loss(self, AB, AB_gc, direction):
        loss_gc = 0.0

        if direction == 0:
            AB_gt = self.rot90(AB_gc.clone().detach(), 1)
            loss_gc = self.criterionGc(AB, AB_gt)
            AB_gc_gt = self.rot90(AB.clone().detach(), 0)
            loss_gc += self.criterionGc(AB_gc, AB_gc_gt)
        else:
            AB_gt = self.rot90(AB_gc.clone().detach(), 0)
            loss_gc = self.criterionGc(AB, AB_gt)
            AB_gc_gt = self.rot90(AB.clone().detach(), 1)
            loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc * self.opt.lambda_AB * self.opt.lambda_gc
        # loss_gc = loss_gc*self.opt.lambda_AB
        return loss_gc

    def get_gc_vf_loss(self, AB, AB_gc):
        loss_gc = 0.0

        size = self.opt.fineSize

        inv_idx = torch.arange(size - 1, -1, -1).long().cuda()

        AB_gt = torch.index_select(AB_gc.clone().detach(), 2, inv_idx)
        loss_gc = self.criterionGc(AB, AB_gt)

        AB_gc_gt = torch.index_select(AB.clone().detach(), 2, inv_idx)
        loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc * self.opt.lambda_AB * self.opt.lambda_gc
        # loss_gc = loss_gc*self.opt.lambda_AB
        return loss_gc

    def get_gc_hf_loss(self, AB, AB_gc):
        loss_gc = 0.0

        size = self.opt.fineSize

        inv_idx = torch.arange(size - 1, -1, -1).long().cuda()

        AB_gt = torch.index_select(AB_gc.clone().detach(), 3, inv_idx)
        loss_gc = self.criterionGc(AB, AB_gt)

        AB_gc_gt = torch.index_select(AB.clone().detach(), 3, inv_idx)
        loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc * self.opt.lambda_AB * self.opt.lambda_gc
        # loss_gc = loss_gc*self.opt.lambda_AB
        return loss_gc

    def backward_D_B(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_gc_B = self.fake_gc_B_pool.query(self.fake_gc_B)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, fake_B, self.netD_gc_B, self.real_gc_B, fake_gc_B)
        self.loss_D_B = loss_D_B.item()

    def backward_G(self):
        # adversariasl loss
        fake_B = self.netG_AB.forward(self.real_A)
        pred_fake = self.netD_B.forward(fake_B)
        loss_G_AB = self.criterionGAN(pred_fake, True) * self.opt.lambda_G

        fake_gc_B = self.netG_AB.forward(self.real_gc_A)
        pred_fake = self.netD_gc_B.forward(fake_gc_B)
        loss_G_gc_AB = self.criterionGAN(pred_fake, True) * self.opt.lambda_G

        if self.opt.geometry == 'rot':
            loss_gc = self.get_gc_rot_loss(fake_B, fake_gc_B, 0)
        elif self.opt.geometry == 'vf':
            loss_gc = self.get_gc_vf_loss(fake_B, fake_gc_B)

        if self.opt.identity > 0:
            # G_AB should be identity if real_B is fed.
            idt_A = self.netG_AB(self.real_B)
            loss_idt = self.criterionIdt(idt_A, self.real_B) * self.opt.lambda_AB * self.opt.identity
            idt_gc_A = self.netG_AB(self.real_gc_B)
            loss_idt_gc = self.criterionIdt(idt_gc_A, self.real_gc_B) * self.opt.lambda_AB * self.opt.identity

            self.idt_A = idt_A.data
            self.idt_gc_A = idt_gc_A.data
            self.loss_idt = loss_idt.item()
            self.loss_idt_gc = loss_idt_gc.item()
        else:
            loss_idt = 0
            loss_idt_gc = 0
            self.loss_idt = 0
            self.loss_idt_gc = 0

        loss_G = loss_G_AB + loss_G_gc_AB + loss_gc + loss_idt + loss_idt_gc

        loss_G.backward()

        self.fake_B = fake_B.data
        self.fake_gc_B = fake_gc_B.data

        self.loss_G_AB = loss_G_AB.item()
        self.loss_G_gc_AB = loss_G_gc_AB.item()
        self.loss_gc = loss_gc.item()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_AB
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_B and D_gc_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_B', self.loss_D_B), ('G_AB', self.loss_G_AB),
                                  ('Gc', self.loss_gc), ('G_gc_AB', self.loss_G_gc_AB)])

        if self.opt.identity > 0.0:
            ret_errors['idt'] = self.loss_idt
            ret_errors['idt_gc'] = self.loss_idt_gc

        return ret_errors

    def get_current_visuals(self):
        real_A = tensor2im(self.real_A.data)
        real_B = tensor2im(self.real_B.data)

        fake_B = tensor2im(self.fake_B)
        fake_gc_B = tensor2im(self.fake_gc_B)

        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B), ('fake_gc_B', fake_gc_B)])
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_AB, 'G_AB', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        self.save_network(self.netD_gc_B, 'D_gc_B', label, self.gpu_ids)

    def test(self):
        # self.netG_AB.eval()
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        input_A = self.input_A.clone()
        input_B = self.input_B.clone()

        size = self.opt.fineSize

        if self.opt.geometry == 'rot':
            self.real_gc_A = self.rot90(input_A, 0)
            self.real_gc_B = self.rot90(input_B, 0)
        elif self.opt.geometry == 'vf':
            inv_idx = torch.arange(size - 1, -1, -1).long().cuda()
            self.real_gc_A = Variable(torch.index_select(input_A, 2, inv_idx))
            self.real_gc_B = Variable(torch.index_select(input_B, 2, inv_idx))
        else:
            raise ValueError("Geometry transformation function [%s] not recognized." % self.opt.geometry)

        self.fake_B = self.netG_AB.forward(self.real_A).data
        self.fake_gc_B = self.netG_AB.forward(self.real_gc_A).data


netG = define_G(3, 3, 64, "resnet_9blocks", "instance", False, "xavier", [0])
gcgan_state_dict = torch.load("latest_net_G_gcgan.pth")
print(gcgan_state_dict.keys())
for i in range(10, 19):
    gcgan_state_dict["model.{}.conv_block.5.weight".format(i)] = gcgan_state_dict["model.{}.conv_block.6.weight".format(i)]
    gcgan_state_dict["model.{}.conv_block.5.bias".format(i)] = gcgan_state_dict["model.{}.conv_block.6.bias".format(i)]
    del gcgan_state_dict["model.{}.conv_block.6.weight".format(i)]
    del gcgan_state_dict["model.{}.conv_block.6.bias".format(i)]
netG.load_state_dict(gcgan_state_dict)
netG.eval()

import os
import cv2
import colour
import glog as log
import kornia

from torch.utils import data
import torchvision.transforms as T
from kornia.losses.psnr import PSNRLoss
from lpips_pytorch import lpips

from datasets.transforms import *
from datasets.iffi import IFFIDataset
from metrics.ssim import SSIM
from utils.data_utils import linear_scaling, linear_unscaling
from configs.default import get_cfg_defaults

opt = get_cfg_defaults()

transform = Compose([
    ResizeTwoInstances(opt.DATASET.SIZE),
    ToTensor(),
    NormalizeTwoInstances((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = IFFIDataset(root=opt.DATASET.TEST_ROOT, transform=transform)
image_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

to_pil = T.ToPILImage()

PSNR = PSNRLoss(max_val=1.)
SSIM = SSIM()

output_dir = "outputs/gcgan-ifr_IFFI_200epochs"
os.makedirs(output_dir, exist_ok=True)


def calc_cie_delta_E(real_path, fake_path):
    real_rgb = cv2.imread(real_path)
    fake_rgb = cv2.imread(fake_path)

    real_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(real_rgb))
    fake_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(fake_rgb))

    delta_E = colour.delta_E(real_lab, fake_lab)
    return np.mean(delta_E)


@torch.no_grad()
def eval():
    psnr_lst, ssim_lst, lpips_lst, deltaE_lst = list(), list(), list(), list()

    # all_preds, all_targets = torch.tensor([]), torch.tensor([])
    for batch_idx, (imgs, y_imgs) in enumerate(image_loader):
        imgs = torch.cat(imgs, dim=0).float().cuda()
        # imgs = linear_scaling(imgs).float().cuda()
        y_imgs = torch.cat(y_imgs, dim=0).float().cuda()
        # y_imgs = y_imgs.float().cuda()
        # y = torch.arange(0, len(self.classes)).cuda()
        # all_targets = torch.cat((all_targets, y.float().cpu()), dim=0)

        output = netG(imgs)
        output = (output.clamp(-1, 1) + 1) / 2.0

        imgs = (imgs + 1) / 2.0
        y_imgs = (y_imgs + 1) / 2.0

        # print(all_preds.size())

        ssim = SSIM(255. * y_imgs, 255. * output).item()
        ssim_lst.append(ssim)

        psnr = -kornia.psnr_loss(output, y_imgs, max_val=1.).item()  # -self.PSNR(y_imgs, output).item()
        psnr_lst.append(psnr)

        lpps = lpips(y_imgs, output, net_type='alex', version='0.1').item() / len(y_imgs)  # TODO ?? not sure working
        lpips_lst.append(lpps)

        # deltaE = colour.delta_E(kornia.rgb_to_lab(y_imgs).permute(0, 2, 3, 1).cpu().numpy(),
        #                         kornia.rgb_to_lab(output).permute(0, 2, 3, 1).detach().cpu().numpy(),
        #                         method='CIE 2000').mean()
        # deltaE_lst.append(deltaE)

        # batch_accuracy = round(torch.mean(torch.tensor(y == y_pred.clone().detach()).float()).item() * 100., 2)

        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        deltaE_batch_lst = list()
        for i, (y_img, img, out) in enumerate(zip(y_imgs.cpu(), imgs.cpu(), output.cpu())):
            real_path = os.path.join(output_dir, "images", "{}_{}_real_A.png".format(batch_idx, i))
            fake_path = os.path.join(output_dir, "images", "{}_{}_real_B.png".format(batch_idx, i))
            filtered_path = os.path.join(output_dir, "images", "{}_{}_fake_B.png".format(batch_idx, i))
            to_pil(y_img).save(real_path)
            to_pil(img).save(filtered_path)
            to_pil(out).save(fake_path)
            deltaE_batch_lst.append(calc_cie_delta_E(real_path, fake_path))
        deltaE = np.mean(deltaE_batch_lst).item()
        deltaE_lst.append(deltaE)
        log.info("{}/{}\tLPIPS: {}\tSSIM: {}\tPSNR: {}\t Delta E: {}"
                 "".format(batch_idx + 1, len(image_loader), round(lpps, 3), round(ssim, 3), round(psnr, 3), round(deltaE, 3)))

    results = {"Dataset": opt.DATASET.NAME, "PSNR": np.mean(psnr_lst), "SSIM": np.mean(ssim_lst), "LPIPS": np.mean(lpips_lst), "Delta_E": np.mean(deltaE_lst)}
    log.info(results)
    # with open(os.path.join(self.output_dir, "metrics.json"), "a+") as f:
    #     json.dump(results, f)


if __name__ == '__main__':
    eval()
