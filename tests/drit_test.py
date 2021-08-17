import torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F


class Dis_content(nn.Module):
    def __init__(self):
        super(Dis_content, self).__init__()
        model = []
        model += [LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(256, 256, kernel_size=4, stride=1, padding=0)]
        model += [nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        out = out.view(-1)
        outs = []
        outs.append(out)
        return outs


class MultiScaleDis(nn.Module):
    def __init__(self, input_dim, n_scale=3, n_layer=4, norm='None', sn=False):
        super(MultiScaleDis, self).__init__()
        ch = 64
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.Diss = nn.ModuleList()
        for _ in range(n_scale):
            self.Diss.append(self._make_net(ch, input_dim, n_layer, norm, sn))

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, 4, 2, 1, norm, sn)]
        tch = ch
        for _ in range(1, n_layer):
            model += [LeakyReLUConv2d(tch, tch * 2, 4, 2, 1, norm, sn)]
            tch *= 2
        if sn:
            model += [spectral_norm(nn.Conv2d(tch, 1, 1, 1, 0))]
        else:
            model += [nn.Conv2d(tch, 1, 1, 1, 0)]
        return nn.Sequential(*model)

    def forward(self, x):
        outs = []
        for Dis in self.Diss:
            outs.append(Dis(x))
            x = self.downsample(x)
        return outs


class Dis(nn.Module):
    def __init__(self, input_dim, norm='None', sn=False):
        super(Dis, self).__init__()
        ch = 64
        n_layer = 6
        self.model = self._make_net(ch, input_dim, n_layer, norm, sn)

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]  # 16
        tch = ch
        for i in range(1, n_layer - 1):
            model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]  # 8
            tch *= 2
        model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm='None', sn=sn)]  # 2
        tch *= 2
        if sn:
            model += [spectral_norm(nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0))]  # 1
        else:
            model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]  # 1
        return nn.Sequential(*model)

    def cuda(self, gpu):
        self.model.cuda(gpu)

    def forward(self, x_A):
        out_A = self.model(x_A)
        out_A = out_A.view(-1)
        outs_A = []
        outs_A.append(out_A)
        return outs_A


####################################################################
# ---------------------------- Encoders -----------------------------
####################################################################
class E_content(nn.Module):
    def __init__(self, input_dim_a, input_dim_b):
        super(E_content, self).__init__()
        encA_c = []
        tch = 64
        encA_c += [LeakyReLUConv2d(input_dim_a, tch, kernel_size=7, stride=1, padding=3)]
        for i in range(1, 3):
            encA_c += [ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
            tch *= 2
        for i in range(0, 3):
            encA_c += [INSResBlock(tch, tch)]

        encB_c = []
        tch = 64
        encB_c += [LeakyReLUConv2d(input_dim_b, tch, kernel_size=7, stride=1, padding=3)]
        for i in range(1, 3):
            encB_c += [ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
            tch *= 2
        for i in range(0, 3):
            encB_c += [INSResBlock(tch, tch)]

        enc_share = []
        for i in range(0, 1):
            enc_share += [INSResBlock(tch, tch)]
            enc_share += [GaussianNoiseLayer()]
            self.conv_share = nn.Sequential(*enc_share)

        self.convA = nn.Sequential(*encA_c)
        self.convB = nn.Sequential(*encB_c)

    def forward(self, xa, xb):
        outputA = self.convA(xa)
        outputB = self.convB(xb)
        outputA = self.conv_share(outputA)
        outputB = self.conv_share(outputB)
        return outputA, outputB

    def forward_a(self, xa):
        outputA = self.convA(xa)
        outputA = self.conv_share(outputA)
        return outputA

    def forward_b(self, xb):
        outputB = self.convB(xb)
        outputB = self.conv_share(outputB)
        return outputB


class E_attr(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, output_nc=8):
        super(E_attr, self).__init__()
        dim = 64
        self.model_a = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_dim_a, dim, 7, 1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim * 2, 4, 2),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim * 2, dim * 4, 4, 2),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim * 4, dim * 4, 4, 2),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim * 4, dim * 4, 4, 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 4, output_nc, 1, 1, 0))
        self.model_b = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_dim_b, dim, 7, 1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim * 2, 4, 2),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim * 2, dim * 4, 4, 2),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim * 4, dim * 4, 4, 2),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim * 4, dim * 4, 4, 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 4, output_nc, 1, 1, 0))
        return

    def forward(self, xa, xb):
        xa = self.model_a(xa)
        xb = self.model_b(xb)
        output_A = xa.view(xa.size(0), -1)
        output_B = xb.view(xb.size(0), -1)
        return output_A, output_B

    def forward_a(self, xa):
        xa = self.model_a(xa)
        output_A = xa.view(xa.size(0), -1)
        return output_A

    def forward_b(self, xb):
        xb = self.model_b(xb)
        output_B = xb.view(xb.size(0), -1)
        return output_B


class E_attr_concat(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, output_nc=8, norm_layer=None, nl_layer=None):
        super(E_attr_concat, self).__init__()

        ndf = 64
        n_blocks = 4
        max_ndf = 4

        conv_layers_A = [nn.ReflectionPad2d(1)]
        conv_layers_A += [nn.Conv2d(input_dim_a, ndf, kernel_size=4, stride=2, padding=0, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
            output_ndf = ndf * min(max_ndf, n + 1)  # 2**n
            conv_layers_A += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
        conv_layers_A += [nl_layer(), nn.AdaptiveAvgPool2d(1)]  # AvgPool2d(13)
        self.fc_A = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.fcVar_A = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv_A = nn.Sequential(*conv_layers_A)

        conv_layers_B = [nn.ReflectionPad2d(1)]
        conv_layers_B += [nn.Conv2d(input_dim_b, ndf, kernel_size=4, stride=2, padding=0, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
            output_ndf = ndf * min(max_ndf, n + 1)  # 2**n
            conv_layers_B += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
        conv_layers_B += [nl_layer(), nn.AdaptiveAvgPool2d(1)]  # AvgPool2d(13)
        self.fc_B = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.fcVar_B = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv_B = nn.Sequential(*conv_layers_B)

    def forward(self, xa, xb):
        x_conv_A = self.conv_A(xa)
        conv_flat_A = x_conv_A.view(xa.size(0), -1)
        output_A = self.fc_A(conv_flat_A)
        outputVar_A = self.fcVar_A(conv_flat_A)
        x_conv_B = self.conv_B(xb)
        conv_flat_B = x_conv_B.view(xb.size(0), -1)
        output_B = self.fc_B(conv_flat_B)
        outputVar_B = self.fcVar_B(conv_flat_B)
        return output_A, outputVar_A, output_B, outputVar_B

    def forward_a(self, xa):
        x_conv_A = self.conv_A(xa)
        conv_flat_A = x_conv_A.view(xa.size(0), -1)
        output_A = self.fc_A(conv_flat_A)
        outputVar_A = self.fcVar_A(conv_flat_A)
        return output_A, outputVar_A

    def forward_b(self, xb):
        x_conv_B = self.conv_B(xb)
        conv_flat_B = x_conv_B.view(xb.size(0), -1)
        output_B = self.fc_B(conv_flat_B)
        outputVar_B = self.fcVar_B(conv_flat_B)
        return output_B, outputVar_B


####################################################################
# --------------------------- Generators ----------------------------
####################################################################
class G(nn.Module):
    def __init__(self, output_dim_a, output_dim_b, nz):
        super(G, self).__init__()
        self.nz = nz
        ini_tch = 256
        tch_add = ini_tch
        tch = ini_tch
        self.tch_add = tch_add
        self.decA1 = MisINSResBlock(tch, tch_add)
        self.decA2 = MisINSResBlock(tch, tch_add)
        self.decA3 = MisINSResBlock(tch, tch_add)
        self.decA4 = MisINSResBlock(tch, tch_add)

        decA5 = []
        decA5 += [ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        tch = tch // 2
        decA5 += [ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        tch = tch // 2
        decA5 += [nn.ConvTranspose2d(tch, output_dim_a, kernel_size=1, stride=1, padding=0)]
        decA5 += [nn.Tanh()]
        self.decA5 = nn.Sequential(*decA5)

        tch = ini_tch
        self.decB1 = MisINSResBlock(tch, tch_add)
        self.decB2 = MisINSResBlock(tch, tch_add)
        self.decB3 = MisINSResBlock(tch, tch_add)
        self.decB4 = MisINSResBlock(tch, tch_add)
        decB5 = []
        decB5 += [ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        tch = tch // 2
        decB5 += [ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        tch = tch // 2
        decB5 += [nn.ConvTranspose2d(tch, output_dim_b, kernel_size=1, stride=1, padding=0)]
        decB5 += [nn.Tanh()]
        self.decB5 = nn.Sequential(*decB5)

        self.mlpA = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, tch_add * 4))
        self.mlpB = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, tch_add * 4))
        return

    def forward_a(self, x, z):
        z = self.mlpA(z)
        z1, z2, z3, z4 = torch.split(z, self.tch_add, dim=1)
        z1, z2, z3, z4 = z1.contiguous(), z2.contiguous(), z3.contiguous(), z4.contiguous()
        out1 = self.decA1(x, z1)
        out2 = self.decA2(out1, z2)
        out3 = self.decA3(out2, z3)
        out4 = self.decA4(out3, z4)
        out = self.decA5(out4)
        return out

    def forward_b(self, x, z):
        z = self.mlpB(z)
        z1, z2, z3, z4 = torch.split(z, self.tch_add, dim=1)
        z1, z2, z3, z4 = z1.contiguous(), z2.contiguous(), z3.contiguous(), z4.contiguous()
        out1 = self.decB1(x, z1)
        out2 = self.decB2(out1, z2)
        out3 = self.decB3(out2, z3)
        out4 = self.decB4(out3, z4)
        out = self.decB5(out4)
        return out


class G_concat(nn.Module):
    def __init__(self, output_dim_a, output_dim_b, nz):
        super(G_concat, self).__init__()
        self.nz = nz
        tch = 256
        dec_share = []
        dec_share += [INSResBlock(tch, tch)]
        self.dec_share = nn.Sequential(*dec_share)
        tch = 256 + self.nz
        decA1 = []
        for i in range(0, 3):
            decA1 += [INSResBlock(tch, tch)]
        tch = tch + self.nz
        decA2 = ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        tch = tch // 2
        tch = tch + self.nz
        decA3 = ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        tch = tch // 2
        tch = tch + self.nz
        decA4 = [nn.ConvTranspose2d(tch, output_dim_a, kernel_size=1, stride=1, padding=0)] + [nn.Tanh()]
        self.decA1 = nn.Sequential(*decA1)
        self.decA2 = nn.Sequential(*[decA2])
        self.decA3 = nn.Sequential(*[decA3])
        self.decA4 = nn.Sequential(*decA4)

        tch = 256 + self.nz
        decB1 = []
        for i in range(0, 3):
            decB1 += [INSResBlock(tch, tch)]
        tch = tch + self.nz
        decB2 = ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        tch = tch // 2
        tch = tch + self.nz
        decB3 = ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        tch = tch // 2
        tch = tch + self.nz
        decB4 = [nn.ConvTranspose2d(tch, output_dim_b, kernel_size=1, stride=1, padding=0)] + [nn.Tanh()]
        self.decB1 = nn.Sequential(*decB1)
        self.decB2 = nn.Sequential(*[decB2])
        self.decB3 = nn.Sequential(*[decB3])
        self.decB4 = nn.Sequential(*decB4)

    def forward_a(self, x, z):
        out0 = self.dec_share(x)
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x_and_z = torch.cat([out0, z_img], 1)
        out1 = self.decA1(x_and_z)
        z_img2 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out1.size(2), out1.size(3))
        x_and_z2 = torch.cat([out1, z_img2], 1)
        out2 = self.decA2(x_and_z2)
        z_img3 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out2.size(2), out2.size(3))
        x_and_z3 = torch.cat([out2, z_img3], 1)
        out3 = self.decA3(x_and_z3)
        z_img4 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out3.size(2), out3.size(3))
        x_and_z4 = torch.cat([out3, z_img4], 1)
        out4 = self.decA4(x_and_z4)
        return out4

    def forward_b(self, x, z):
        out0 = self.dec_share(x)
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x_and_z = torch.cat([out0, z_img], 1)
        out1 = self.decB1(x_and_z)
        z_img2 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out1.size(2), out1.size(3))
        x_and_z2 = torch.cat([out1, z_img2], 1)
        out2 = self.decB2(x_and_z2)
        z_img3 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out2.size(2), out2.size(3))
        x_and_z3 = torch.cat([out2, z_img3], 1)
        out3 = self.decB3(x_and_z3)
        z_img4 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out3.size(2), out3.size(3))
        x_and_z4 = torch.cat([out3, z_img4], 1)
        out4 = self.decB4(x_and_z4)
        return out4


####################################################################
# ------------------------- Basic Functions -------------------------
####################################################################
def get_scheduler(optimizer, opts, cur_ep=-1):
    if opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
    else:
        return NotImplementedError('no such learn rate policy')
    return scheduler


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += conv3x3(inplanes, outplanes)
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


def get_norm_layer(layer_type='instance'):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=False)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


def conv3x3(in_planes, out_planes):
    return [nn.ReflectionPad2d(1), nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=True)]


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)


####################################################################
# -------------------------- Basic Blocks --------------------------
####################################################################

## The code of LayerNorm is modified from MUNIT (https://github.com/NVlabs/MUNIT)
class LayerNorm(nn.Module):
    def __init__(self, n_out, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.n_out = n_out
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
            self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
        return

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
        else:
            return F.layer_norm(x, normalized_shape)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += conv3x3(inplanes, inplanes)
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class LeakyReLUConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            model += [spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if 'norm' == 'Instance':
            model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        # elif == 'Group'

    def forward(self, x):
        return self.model(x)


class ReLUINSConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(ReLUINSConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


class INSResBlock(nn.Module):
    def conv3x3(self, inplanes, out_planes, stride=1):
        return [nn.ReflectionPad2d(1), nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride)]

    def __init__(self, inplanes, planes, stride=1, dropout=0.0):
        super(INSResBlock, self).__init__()
        model = []
        model += self.conv3x3(inplanes, planes, stride)
        model += [nn.InstanceNorm2d(planes)]
        model += [nn.ReLU(inplace=True)]
        model += self.conv3x3(planes, planes)
        model += [nn.InstanceNorm2d(planes)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class MisINSResBlock(nn.Module):
    def conv3x3(self, dim_in, dim_out, stride=1):
        return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride))

    def conv1x1(self, dim_in, dim_out):
        return nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)

    def __init__(self, dim, dim_extra, stride=1, dropout=0.0):
        super(MisINSResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            self.conv3x3(dim, dim, stride),
            nn.InstanceNorm2d(dim))
        self.conv2 = nn.Sequential(
            self.conv3x3(dim, dim, stride),
            nn.InstanceNorm2d(dim))
        self.blk1 = nn.Sequential(
            self.conv1x1(dim + dim_extra, dim + dim_extra),
            nn.ReLU(inplace=False),
            self.conv1x1(dim + dim_extra, dim),
            nn.ReLU(inplace=False))
        self.blk2 = nn.Sequential(
            self.conv1x1(dim + dim_extra, dim + dim_extra),
            nn.ReLU(inplace=False),
            self.conv1x1(dim + dim_extra, dim),
            nn.ReLU(inplace=False))
        model = []
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.blk1.apply(gaussian_weights_init)
        self.blk2.apply(gaussian_weights_init)

    def forward(self, x, z):
        residual = x
        z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        o1 = self.conv1(x)
        o2 = self.blk1(torch.cat([o1, z_expand], dim=1))
        o3 = self.conv2(o2)
        out = self.blk2(torch.cat([o3, z_expand], dim=1))
        out += residual
        return out


class GaussianNoiseLayer(nn.Module):
    def __init__(self, ):
        super(GaussianNoiseLayer, self).__init__()

    def forward(self, x):
        if self.training == False:
            return x
        noise = Variable(torch.randn(x.size()).cuda(x.get_device()))
        return x + noise


class ReLUINSConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
        super(ReLUINSConvTranspose2d, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)]
        model += [LayerNorm(n_out)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


####################################################################
# --------------------- Spectral Normalization ---------------------
#  This part of code is copied from pytorch master branch (0.5.0)
####################################################################
class SpectralNorm(object):
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        weight_mat = weight_mat.reshape(height, -1)
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
        sigma = torch.dot(u, torch.matmul(weight_mat, v))
        weight = weight / sigma
        return weight, u

    def remove(self, module):
        weight = getattr(module, self.name)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight))

    def __call__(self, module, inputs):
        if module.training:
            weight, u = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, self.name + '_u', u)
        else:
            r_g = getattr(module, self.name + '_orig').requires_grad
            getattr(module, self.name).detach_().requires_grad_(r_g)

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        height = weight.size(dim)
        u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_forward_pre_hook(fn)
        return fn


def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


def remove_spectral_norm(module, name='weight'):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module
    raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))


class DRIT(nn.Module):
    def __init__(self, opts):
        super(DRIT, self).__init__()

        # parameters
        lr = 0.0001
        lr_dcontent = lr / 2.5
        self.nz = 8
        self.concat = opts.concat
        self.no_ms = opts.no_ms
        self.gpu = opts.gpu

        # discriminators
        if opts.dis_scale > 1:
            self.disA = MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
            self.disB = MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
            self.disA2 = MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
            self.disB2 = MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
        else:
            self.disA = Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
            self.disB = Dis(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
            self.disA2 = Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
            self.disB2 = Dis(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
        self.disContent = Dis_content()

        # encoders
        self.enc_c = E_content(opts.input_dim_a, opts.input_dim_b)
        if self.concat:
            self.enc_a = E_attr_concat(opts.input_dim_a, opts.input_dim_b, self.nz, norm_layer=None, nl_layer=get_non_linearity(layer_type='lrelu'))
        else:
            self.enc_a = E_attr(opts.input_dim_a, opts.input_dim_b, self.nz)

        # generator
        if self.concat:
            self.gen = G_concat(opts.input_dim_a, opts.input_dim_b, nz=self.nz)
        else:
            self.gen = G(opts.input_dim_a, opts.input_dim_b, nz=self.nz)

        # optimizers
        self.disA_opt = torch.optim.Adam(self.disA.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disB_opt = torch.optim.Adam(self.disB.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disA2_opt = torch.optim.Adam(self.disA2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disB2_opt = torch.optim.Adam(self.disB2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disContent_opt = torch.optim.Adam(self.disContent.parameters(), lr=lr_dcontent, betas=(0.5, 0.999), weight_decay=0.0001)
        self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

        # Setup the loss function for training
        self.criterionL1 = torch.nn.L1Loss()

    def initialize(self):
        self.disA.apply(gaussian_weights_init)
        self.disB.apply(gaussian_weights_init)
        self.disA2.apply(gaussian_weights_init)
        self.disB2.apply(gaussian_weights_init)
        self.disContent.apply(gaussian_weights_init)
        self.gen.apply(gaussian_weights_init)
        self.enc_c.apply(gaussian_weights_init)
        self.enc_a.apply(gaussian_weights_init)

    def set_scheduler(self, opts, last_ep=0):
        self.disA_sch = get_scheduler(self.disA_opt, opts, last_ep)
        self.disB_sch = get_scheduler(self.disB_opt, opts, last_ep)
        self.disA2_sch = get_scheduler(self.disA2_opt, opts, last_ep)
        self.disB2_sch = get_scheduler(self.disB2_opt, opts, last_ep)
        self.disContent_sch = get_scheduler(self.disContent_opt, opts, last_ep)
        self.enc_c_sch = get_scheduler(self.enc_c_opt, opts, last_ep)
        self.enc_a_sch = get_scheduler(self.enc_a_opt, opts, last_ep)
        self.gen_sch = get_scheduler(self.gen_opt, opts, last_ep)

    def setgpu(self, gpu):
        self.gpu = gpu
        self.disA.cuda(self.gpu)
        self.disB.cuda(self.gpu)
        self.disA2.cuda(self.gpu)
        self.disB2.cuda(self.gpu)
        self.disContent.cuda(self.gpu)
        self.enc_c.cuda(self.gpu)
        self.enc_a.cuda(self.gpu)
        self.gen.cuda(self.gpu)

    def get_z_random(self, batchSize, nz, random_type='gauss'):
        z = torch.randn(batchSize, nz).cuda(self.gpu)
        return z

    def test_forward(self, image, a2b=True):
        self.z_random = self.get_z_random(image.size(0), self.nz, 'gauss')
        if a2b:
            self.z_content = self.enc_c.forward_a(image)
            output = self.gen.forward_b(self.z_content, self.z_random)
        else:
            self.z_content = self.enc_c.forward_b(image)
            output = self.gen.forward_a(self.z_content, self.z_random)
        return output

    def test_forward_transfer(self, image_a, image_b, a2b=True):
        self.z_content_a, self.z_content_b = self.enc_c.forward(image_a, image_b)
        if self.concat:
            self.mu_a, self.logvar_a, self.mu_b, self.logvar_b = self.enc_a.forward(image_a, image_b)
            std_a = self.logvar_a.mul(0.5).exp_()
            eps = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
            self.z_attr_a = eps.mul(std_a).add_(self.mu_a)
            std_b = self.logvar_b.mul(0.5).exp_()
            eps = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
            self.z_attr_b = eps.mul(std_b).add_(self.mu_b)
        else:
            self.z_attr_a, self.z_attr_b = self.enc_a.forward(image_a, image_b)
        if a2b:
            output = self.gen.forward_b(self.z_content_a, self.z_attr_b)
        else:
            output = self.gen.forward_a(self.z_content_b, self.z_attr_a)
        return output

    def forward(self):
        # input images
        half_size = 1
        real_A = self.input_A
        real_B = self.input_B
        self.real_A_encoded = real_A[0:half_size]
        self.real_A_random = real_A[half_size:]
        self.real_B_encoded = real_B[0:half_size]
        self.real_B_random = real_B[half_size:]

        # get encoded z_c
        self.z_content_a, self.z_content_b = self.enc_c.forward(self.real_A_encoded, self.real_B_encoded)

        # get encoded z_a
        if self.concat:
            self.mu_a, self.logvar_a, self.mu_b, self.logvar_b = self.enc_a.forward(self.real_A_encoded, self.real_B_encoded)
            std_a = self.logvar_a.mul(0.5).exp_()
            eps_a = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
            self.z_attr_a = eps_a.mul(std_a).add_(self.mu_a)
            std_b = self.logvar_b.mul(0.5).exp_()
            eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
            self.z_attr_b = eps_b.mul(std_b).add_(self.mu_b)
        else:
            self.z_attr_a, self.z_attr_b = self.enc_a.forward(self.real_A_encoded, self.real_B_encoded)

        # get random z_a
        self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.nz, 'gauss')
        if not self.no_ms:
            self.z_random2 = self.get_z_random(self.real_A_encoded.size(0), self.nz, 'gauss')

        # first cross translation
        if not self.no_ms:
            input_content_forA = torch.cat((self.z_content_b, self.z_content_a, self.z_content_b, self.z_content_b), 0)
            input_content_forB = torch.cat((self.z_content_a, self.z_content_b, self.z_content_a, self.z_content_a), 0)
            input_attr_forA = torch.cat((self.z_attr_a, self.z_attr_a, self.z_random, self.z_random2), 0)
            input_attr_forB = torch.cat((self.z_attr_b, self.z_attr_b, self.z_random, self.z_random2), 0)
            output_fakeA = self.gen.forward_a(input_content_forA, input_attr_forA)
            output_fakeB = self.gen.forward_b(input_content_forB, input_attr_forB)
            self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random, self.fake_A_random2 = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
            self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random, self.fake_B_random2 = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)
        else:
            input_content_forA = torch.cat((self.z_content_b, self.z_content_a, self.z_content_b), 0)
            input_content_forB = torch.cat((self.z_content_a, self.z_content_b, self.z_content_a), 0)
            input_attr_forA = torch.cat((self.z_attr_a, self.z_attr_a, self.z_random), 0)
            input_attr_forB = torch.cat((self.z_attr_b, self.z_attr_b, self.z_random), 0)
            output_fakeA = self.gen.forward_a(input_content_forA, input_attr_forA)
            output_fakeB = self.gen.forward_b(input_content_forB, input_attr_forB)
            self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
            self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)

        # get reconstructed encoded z_c
        self.z_content_recon_b, self.z_content_recon_a = self.enc_c.forward(self.fake_A_encoded, self.fake_B_encoded)

        # get reconstructed encoded z_a
        if self.concat:
            self.mu_recon_a, self.logvar_recon_a, self.mu_recon_b, self.logvar_recon_b = self.enc_a.forward(self.fake_A_encoded, self.fake_B_encoded)
            std_a = self.logvar_recon_a.mul(0.5).exp_()
            eps_a = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
            self.z_attr_recon_a = eps_a.mul(std_a).add_(self.mu_recon_a)
            std_b = self.logvar_recon_b.mul(0.5).exp_()
            eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
            self.z_attr_recon_b = eps_b.mul(std_b).add_(self.mu_recon_b)
        else:
            self.z_attr_recon_a, self.z_attr_recon_b = self.enc_a.forward(self.fake_A_encoded, self.fake_B_encoded)

        # second cross translation
        self.fake_A_recon = self.gen.forward_a(self.z_content_recon_a, self.z_attr_recon_a)
        self.fake_B_recon = self.gen.forward_b(self.z_content_recon_b, self.z_attr_recon_b)

        # for display
        self.image_display = torch.cat((self.real_A_encoded[0:1].detach().cpu(), self.fake_B_encoded[0:1].detach().cpu(), \
                                        self.fake_B_random[0:1].detach().cpu(), self.fake_AA_encoded[0:1].detach().cpu(), self.fake_A_recon[0:1].detach().cpu(), \
                                        self.real_B_encoded[0:1].detach().cpu(), self.fake_A_encoded[0:1].detach().cpu(), \
                                        self.fake_A_random[0:1].detach().cpu(), self.fake_BB_encoded[0:1].detach().cpu(), self.fake_B_recon[0:1].detach().cpu()), dim=0)

        # for latent regression
        if self.concat:
            self.mu2_a, _, self.mu2_b, _ = self.enc_a.forward(self.fake_A_random, self.fake_B_random)
        else:
            self.z_attr_random_a, self.z_attr_random_b = self.enc_a.forward(self.fake_A_random, self.fake_B_random)

    def forward_content(self):
        half_size = 1
        self.real_A_encoded = self.input_A[0:half_size]
        self.real_B_encoded = self.input_B[0:half_size]
        # get encoded z_c
        self.z_content_a, self.z_content_b = self.enc_c.forward(self.real_A_encoded, self.real_B_encoded)

    def update_D_content(self, image_a, image_b):
        self.input_A = image_a
        self.input_B = image_b
        self.forward_content()
        self.disContent_opt.zero_grad()
        loss_D_Content = self.backward_contentD(self.z_content_a, self.z_content_b)
        self.disContent_loss = loss_D_Content.item()
        nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
        self.disContent_opt.step()

    def update_D(self, image_a, image_b):
        self.input_A = image_a
        self.input_B = image_b
        self.forward()

        # update disA
        self.disA_opt.zero_grad()
        loss_D1_A = self.backward_D(self.disA, self.real_A_encoded, self.fake_A_encoded)
        self.disA_loss = loss_D1_A.item()
        self.disA_opt.step()

        # update disA2
        self.disA2_opt.zero_grad()
        loss_D2_A = self.backward_D(self.disA2, self.real_A_random, self.fake_A_random)
        self.disA2_loss = loss_D2_A.item()
        if not self.no_ms:
            loss_D2_A2 = self.backward_D(self.disA2, self.real_A_random, self.fake_A_random2)
            self.disA2_loss += loss_D2_A2.item()
        self.disA2_opt.step()

        # update disB
        self.disB_opt.zero_grad()
        loss_D1_B = self.backward_D(self.disB, self.real_B_encoded, self.fake_B_encoded)
        self.disB_loss = loss_D1_B.item()
        self.disB_opt.step()

        # update disB2
        self.disB2_opt.zero_grad()
        loss_D2_B = self.backward_D(self.disB2, self.real_B_random, self.fake_B_random)
        self.disB2_loss = loss_D2_B.item()
        if not self.no_ms:
            loss_D2_B2 = self.backward_D(self.disB2, self.real_B_random, self.fake_B_random2)
            self.disB2_loss += loss_D2_B2.item()
        self.disB2_opt.step()

        # update disContent
        self.disContent_opt.zero_grad()
        loss_D_Content = self.backward_contentD(self.z_content_a, self.z_content_b)
        self.disContent_loss = loss_D_Content.item()
        nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
        self.disContent_opt.step()

    def backward_D(self, netD, real, fake):
        pred_fake = netD.forward(fake.detach())
        pred_real = netD.forward(real)
        loss_D = 0
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = nn.functional.sigmoid(out_a)
            out_real = nn.functional.sigmoid(out_b)
            all0 = torch.zeros_like(out_fake).cuda(self.gpu)
            all1 = torch.ones_like(out_real).cuda(self.gpu)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            loss_D += ad_true_loss + ad_fake_loss
        loss_D.backward()
        return loss_D

    def backward_contentD(self, imageA, imageB):
        pred_fake = self.disContent.forward(imageA.detach())
        pred_real = self.disContent.forward(imageB.detach())
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = nn.functional.sigmoid(out_a)
            out_real = nn.functional.sigmoid(out_b)
            all1 = torch.ones((out_real.size(0))).cuda(self.gpu)
            all0 = torch.zeros((out_fake.size(0))).cuda(self.gpu)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
        loss_D = ad_true_loss + ad_fake_loss
        loss_D.backward()
        return loss_D

    def update_EG(self):
        # update G, Ec, Ea
        self.enc_c_opt.zero_grad()
        self.enc_a_opt.zero_grad()
        self.gen_opt.zero_grad()
        self.backward_EG()
        self.enc_c_opt.step()
        self.enc_a_opt.step()
        self.gen_opt.step()

        # update G, Ec
        self.enc_c_opt.zero_grad()
        self.gen_opt.zero_grad()
        self.backward_G_alone()
        self.enc_c_opt.step()
        self.gen_opt.step()

    def backward_EG(self):
        # content Ladv for generator
        loss_G_GAN_Acontent = self.backward_G_GAN_content(self.z_content_a)
        loss_G_GAN_Bcontent = self.backward_G_GAN_content(self.z_content_b)

        # Ladv for generator
        loss_G_GAN_A = self.backward_G_GAN(self.fake_A_encoded, self.disA)
        loss_G_GAN_B = self.backward_G_GAN(self.fake_B_encoded, self.disB)

        # KL loss - z_a
        if self.concat:
            kl_element_a = self.mu_a.pow(2).add_(self.logvar_a.exp()).mul_(-1).add_(1).add_(self.logvar_a)
            loss_kl_za_a = torch.sum(kl_element_a).mul_(-0.5) * 0.01
            kl_element_b = self.mu_b.pow(2).add_(self.logvar_b.exp()).mul_(-1).add_(1).add_(self.logvar_b)
            loss_kl_za_b = torch.sum(kl_element_b).mul_(-0.5) * 0.01
        else:
            loss_kl_za_a = self._l2_regularize(self.z_attr_a) * 0.01
            loss_kl_za_b = self._l2_regularize(self.z_attr_b) * 0.01

        # KL loss - z_c
        loss_kl_zc_a = self._l2_regularize(self.z_content_a) * 0.01
        loss_kl_zc_b = self._l2_regularize(self.z_content_b) * 0.01

        # cross cycle consistency loss
        loss_G_L1_A = self.criterionL1(self.fake_A_recon, self.real_A_encoded) * 10
        loss_G_L1_B = self.criterionL1(self.fake_B_recon, self.real_B_encoded) * 10
        loss_G_L1_AA = self.criterionL1(self.fake_AA_encoded, self.real_A_encoded) * 10
        loss_G_L1_BB = self.criterionL1(self.fake_BB_encoded, self.real_B_encoded) * 10

        loss_G = loss_G_GAN_A + loss_G_GAN_B + \
                 loss_G_GAN_Acontent + loss_G_GAN_Bcontent + \
                 loss_G_L1_AA + loss_G_L1_BB + \
                 loss_G_L1_A + loss_G_L1_B + \
                 loss_kl_zc_a + loss_kl_zc_b + \
                 loss_kl_za_a + loss_kl_za_b

        loss_G.backward(retain_graph=True)

        self.gan_loss_a = loss_G_GAN_A.item()
        self.gan_loss_b = loss_G_GAN_B.item()
        self.gan_loss_acontent = loss_G_GAN_Acontent.item()
        self.gan_loss_bcontent = loss_G_GAN_Bcontent.item()
        self.kl_loss_za_a = loss_kl_za_a.item()
        self.kl_loss_za_b = loss_kl_za_b.item()
        self.kl_loss_zc_a = loss_kl_zc_a.item()
        self.kl_loss_zc_b = loss_kl_zc_b.item()
        self.l1_recon_A_loss = loss_G_L1_A.item()
        self.l1_recon_B_loss = loss_G_L1_B.item()
        self.l1_recon_AA_loss = loss_G_L1_AA.item()
        self.l1_recon_BB_loss = loss_G_L1_BB.item()
        self.G_loss = loss_G.item()

    def backward_G_GAN_content(self, data):
        outs = self.disContent.forward(data)
        for out in outs:
            outputs_fake = nn.functional.sigmoid(out)
            all_half = 0.5 * torch.ones((outputs_fake.size(0))).cuda(self.gpu)
            ad_loss = nn.functional.binary_cross_entropy(outputs_fake, all_half)
        return ad_loss

    def backward_G_GAN(self, fake, netD=None):
        outs_fake = netD.forward(fake)
        loss_G = 0
        for out_a in outs_fake:
            outputs_fake = nn.functional.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
            loss_G += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
        return loss_G

    def backward_G_alone(self):
        # Ladv for generator
        loss_G_GAN2_A = self.backward_G_GAN(self.fake_A_random, self.disA2)
        loss_G_GAN2_B = self.backward_G_GAN(self.fake_B_random, self.disB2)
        if not self.no_ms:
            loss_G_GAN2_A2 = self.backward_G_GAN(self.fake_A_random2, self.disA2)
            loss_G_GAN2_B2 = self.backward_G_GAN(self.fake_B_random2, self.disB2)

        # mode seeking loss for A-->B and B-->A
        if not self.no_ms:
            lz_AB = torch.mean(torch.abs(self.fake_B_random2 - self.fake_B_random)) / torch.mean(torch.abs(self.z_random2 - self.z_random))
            lz_BA = torch.mean(torch.abs(self.fake_A_random2 - self.fake_A_random)) / torch.mean(torch.abs(self.z_random2 - self.z_random))
            eps = 1 * 1e-5
            loss_lz_AB = 1 / (lz_AB + eps)
            loss_lz_BA = 1 / (lz_BA + eps)
        # latent regression loss
        if self.concat:
            loss_z_L1_a = torch.mean(torch.abs(self.mu2_a - self.z_random)) * 10
            loss_z_L1_b = torch.mean(torch.abs(self.mu2_b - self.z_random)) * 10
        else:
            loss_z_L1_a = torch.mean(torch.abs(self.z_attr_random_a - self.z_random)) * 10
            loss_z_L1_b = torch.mean(torch.abs(self.z_attr_random_b - self.z_random)) * 10

        loss_z_L1 = loss_z_L1_a + loss_z_L1_b + loss_G_GAN2_A + loss_G_GAN2_B
        if not self.no_ms:
            loss_z_L1 += (loss_G_GAN2_A2 + loss_G_GAN2_B2)
            loss_z_L1 += (loss_lz_AB + loss_lz_BA)
        loss_z_L1.backward()
        self.l1_recon_z_loss_a = loss_z_L1_a.item()
        self.l1_recon_z_loss_b = loss_z_L1_b.item()
        if not self.no_ms:
            self.gan2_loss_a = loss_G_GAN2_A.item() + loss_G_GAN2_A2.item()
            self.gan2_loss_b = loss_G_GAN2_B.item() + loss_G_GAN2_B2.item()
            self.lz_AB = loss_lz_AB.item()
            self.lz_BA = loss_lz_BA.item()
        else:
            self.gan2_loss_a = loss_G_GAN2_A.item()
            self.gan2_loss_b = loss_G_GAN2_B.item()

    def update_lr(self):
        self.disA_sch.step()
        self.disB_sch.step()
        self.disA2_sch.step()
        self.disB2_sch.step()
        self.disContent_sch.step()
        self.enc_c_sch.step()
        self.enc_a_sch.step()
        self.gen_sch.step()

    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir, map_location=torch.device("cuda"))
        # weight
        if train:
            self.disA.load_state_dict(checkpoint['disA'])
            self.disA2.load_state_dict(checkpoint['disA2'])
            self.disB.load_state_dict(checkpoint['disB'])
            self.disB2.load_state_dict(checkpoint['disB2'])
            self.disContent.load_state_dict(checkpoint['disContent'])
        self.enc_c.load_state_dict(checkpoint['enc_c'])
        self.enc_a.load_state_dict(checkpoint['enc_a'])
        self.gen.load_state_dict(checkpoint['gen'])
        # optimizer
        if train:
            self.disA_opt.load_state_dict(checkpoint['disA_opt'])
            self.disA2_opt.load_state_dict(checkpoint['disA2_opt'])
            self.disB_opt.load_state_dict(checkpoint['disB_opt'])
            self.disB2_opt.load_state_dict(checkpoint['disB2_opt'])
            self.disContent_opt.load_state_dict(checkpoint['disContent_opt'])
            self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
            self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
            self.gen_opt.load_state_dict(checkpoint['gen_opt'])
        return checkpoint['ep'], checkpoint['total_it']

    def save(self, filename, ep, total_it):
        state = {
            'disA': self.disA.state_dict(),
            'disA2': self.disA2.state_dict(),
            'disB': self.disB.state_dict(),
            'disB2': self.disB2.state_dict(),
            'disContent': self.disContent.state_dict(),
            'enc_c': self.enc_c.state_dict(),
            'enc_a': self.enc_a.state_dict(),
            'gen': self.gen.state_dict(),
            'disA_opt': self.disA_opt.state_dict(),
            'disA2_opt': self.disA2_opt.state_dict(),
            'disB_opt': self.disB_opt.state_dict(),
            'disB2_opt': self.disB2_opt.state_dict(),
            'disContent_opt': self.disContent_opt.state_dict(),
            'enc_c_opt': self.enc_c_opt.state_dict(),
            'enc_a_opt': self.enc_a_opt.state_dict(),
            'gen_opt': self.gen_opt.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        torch.save(state, filename)
        return

    def assemble_outputs(self):
        images_a = self.normalize_image(self.real_A_encoded).detach()
        images_b = self.normalize_image(self.real_B_encoded).detach()
        images_a1 = self.normalize_image(self.fake_A_encoded).detach()
        images_a2 = self.normalize_image(self.fake_A_random).detach()
        images_a3 = self.normalize_image(self.fake_A_recon).detach()
        images_a4 = self.normalize_image(self.fake_AA_encoded).detach()
        images_b1 = self.normalize_image(self.fake_B_encoded).detach()
        images_b2 = self.normalize_image(self.fake_B_random).detach()
        images_b3 = self.normalize_image(self.fake_B_recon).detach()
        images_b4 = self.normalize_image(self.fake_BB_encoded).detach()
        row1 = torch.cat((images_a[0:1, ::], images_b1[0:1, ::], images_b2[0:1, ::], images_a4[0:1, ::], images_a3[0:1, ::]), 3)
        row2 = torch.cat((images_b[0:1, ::], images_a1[0:1, ::], images_a2[0:1, ::], images_b4[0:1, ::], images_b3[0:1, ::]), 3)
        return torch.cat((row1, row2), 2)

    def normalize_image(self, x):
        return x[:, 0:3, :, :]


import argparse


class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data loader related
        self.parser.add_argument('--dataroot', type=str, required=True, help='path of data')
        self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
        self.parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
        self.parser.add_argument('--crop_size', type=int, default=216, help='cropped image size for training')
        self.parser.add_argument('--nThreads', type=int, default=4, help='for data loader')
        self.parser.add_argument('--input_dim_a', type=int, default=3, help='# of input channels for domain A')
        self.parser.add_argument('--input_dim_b', type=int, default=3, help='# of input channels for domain B')
        self.parser.add_argument('--a2b', type=int, default=1, help='translation direction, 1 for a2b, 0 for b2a')

        # ouptput related
        self.parser.add_argument('--num', type=int, default=5, help='number of outputs per image')
        self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
        self.parser.add_argument('--result_dir', type=str, default='../outputs', help='path for saving result images and models')

        # model related
        self.parser.add_argument('--concat', type=int, default=1, help='concatenate attribute features for translation, set 0 for using feature-wise transform')
        self.parser.add_argument('--no_ms', action='store_true', help='disable mode seeking regularization')
        self.parser.add_argument('--resume', type=str, required=True, help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        # set irrelevant options
        self.opt.dis_scale = 3
        self.opt.dis_norm = 'None'
        self.opt.dis_spectral_norm = False
        return self.opt


parser = TestOptions()
opts = parser.parse()
model = DRIT(opts)
model.resume("latest_net_G_drit_plus.pth", train=False)
model.setgpu(0)
model.eval()

import os
import cv2
import colour
import glog as log
import kornia
import numpy as np

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

output_dir = "outputs/drit-ifr_IFFI_200epochs"
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

        z_random = model.get_z_random(imgs.size(0), 8, 'gauss')
        z_content = model.enc_c.forward_a(imgs)
        output = model.gen.forward_b(z_content, z_random)
        output = (output.clamp(-1, 1) + 1) / 2.0

        imgs = (imgs + 1) / 2.0
        y_imgs = (y_imgs + 1) / 2.0

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
