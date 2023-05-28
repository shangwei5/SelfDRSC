import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import warnings
from torch.autograd import Variable
from thop import profile
import math
try:
    from .warplayer import multi_warp
except:
    from model.warplayer import multi_warp
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from models import flow_pwc
# from data.distortion_prior import distortion_map

def generate_2D_grid(H, W):
    x = torch.arange(0, W, 1).float()
    y = torch.arange(0, H, 1).float()

    xx = x.repeat(H, 1)
    yy = y.view(H, 1).repeat(1, W)

    grid = torch.stack([xx, yy], dim=0)

    return grid  # (2,H,W)


def time_map(h, w, ref_row, reverse=False):
    grid_row = generate_2D_grid(h, w)[1].float()
    mask = grid_row / ref_row
    formask = mask.clone()
    backmask = mask.clone()
    warpmask = mask.clone()

    warpmask[warpmask <= 1] = 1
    warpmask[warpmask > 1] = 0

    if reverse:
        formask[formask > 1] = 1
        formask = torch.flip(formask, [0])

        backmask[backmask < 1] = -1
        backmask[backmask >= 1] -= 1
        backmask = backmask / backmask.max()
        backmask[backmask < 0] = 0
        backmask = torch.flip(backmask, [0])
        warpmask = torch.flip(warpmask, [0])

    else:
        formask[formask > 1] = 1
        '''
        0,0,0,
        1/ref_row, 1/ref_row, 1/ref_row
        ...
        1,1,1
        ...
        1,1,1
        '''
        backmask[backmask < 1] = 0
        backmask[backmask >= 1] -= 1
        backmask = backmask / backmask.max()
        '''
            0,0,0,
            ...
            0,0,0
            1/(h-ref_row), 1/(h-ref_row), 1/(h-ref_row)
            ...
            1,1,1
        '''

    return formask[..., np.newaxis], backmask[..., np.newaxis], warpmask[..., np.newaxis]

def pad(img, ratio=32):
    if len(img.shape) == 5:
        b, n, c, h, w = img.shape
        img = img.reshape(b * n, c, h, w)
        ph = ((h - 1) // ratio + 1) * ratio
        pw = ((w - 1) // ratio + 1) * ratio
        padding = (0, pw - w, 0, ph - h)
        img = F.pad(img, padding, mode='replicate')
        img = img.reshape(b, n, c, ph, pw)
        return img
    elif len(img.shape) == 4:
        n, c, h, w = img.shape
        ph = ((h - 1) // ratio + 1) * ratio
        pw = ((w - 1) // ratio + 1) * ratio
        padding = (0, pw - w, 0, ph - h)
        img = F.pad(img, padding, mode='replicate')
        return img

def CFR_flow_t_align(device, flow_01, flow_10, t_value):
    """ modified from https://github.com/JihyongOh/XVFI/blob/main/XVFInet.py"""
    ## Feature warping
    flow_01, norm0 = fwarp(device, flow_01,
                           t_value * flow_01)  ## Actually, F (t) -> (t+1). Translation. Not normalized yet
    flow_10, norm1 = fwarp(device, flow_10, (
            1 - t_value) * flow_10)  ## Actually, F (1-t) -> (-t). Translation. Not normalized yet

    flow_t0 = -(1 - t_value) * (t_value) * flow_01 + (t_value) * (t_value) * flow_10
    flow_t1 = (1 - t_value) * (1 - t_value) * flow_01 - (t_value) * (1 - t_value) * flow_10

    norm = (1 - t_value) * norm0 + t_value * norm1
    mask_ = (norm.detach() > 0).type(norm.type())
    flow_t0 = (1 - mask_) * flow_t0 + mask_ * (flow_t0.clone() / (norm.clone() + (1 - mask_)))
    flow_t1 = (1 - mask_) * flow_t1 + mask_ * (flow_t1.clone() / (norm.clone() + (1 - mask_)))

    return flow_t0, flow_t1

def fwarp(device, img, flo):
    """
        -img: image (N, C, H, W)
        -flo: optical flow (N, 2, H, W)
        elements of flo is in [0, H] and [0, W] for dx, dy

    """

    # (x1, y1)		(x1, y2)
    # +---------------+
    # |				  |
    # |	o(x, y) 	  |
    # |				  |
    # |				  |
    # |				  |
    # |				  |
    # +---------------+
    # (x2, y1)		(x2, y2)

    N, C, _, _ = img.size()

    # translate start-point optical flow to end-point optical flow
    y = flo[:, 0:1:, :]
    x = flo[:, 1:2, :, :]

    x = x.repeat(1, C, 1, 1)
    y = y.repeat(1, C, 1, 1)

    # Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
    x1 = torch.floor(x)
    x2 = x1 + 1
    y1 = torch.floor(y)
    y2 = y1 + 1

    # firstly, get gaussian weights
    w11, w12, w21, w22 = get_gaussian_weights(x, y, x1, x2, y1, y2)

    # secondly, sample each weighted corner
    img11, o11 = sample_one(device, img, x1, y1, w11)
    img12, o12 = sample_one(device, img, x1, y2, w12)
    img21, o21 = sample_one(device, img, x2, y1, w21)
    img22, o22 = sample_one(device, img, x2, y2, w22)

    imgw = img11 + img12 + img21 + img22
    o = o11 + o12 + o21 + o22

    return imgw, o

def get_gaussian_weights(x, y, x1, x2, y1, y2):
    w11 = torch.exp(-((x - x1) ** 2 + (y - y1) ** 2))
    w12 = torch.exp(-((x - x1) ** 2 + (y - y2) ** 2))
    w21 = torch.exp(-((x - x2) ** 2 + (y - y1) ** 2))
    w22 = torch.exp(-((x - x2) ** 2 + (y - y2) ** 2))

    return w11, w12, w21, w22


def sample_one(device, img, shiftx, shifty, weight):
    """
    Input:
        -img (N, C, H, W)
        -shiftx, shifty (N, c, H, W)
    """

    N, C, H, W = img.size()

    # flatten all (all restored as Tensors)
    flat_shiftx = shiftx.view(-1)
    flat_shifty = shifty.view(-1)
    flat_basex = torch.arange(0, H, requires_grad=False).view(-1, 1)[None, None].to(device).long().repeat(N, C,
                                                                                                          1,
                                                                                                          W).view(
        -1)
    flat_basey = torch.arange(0, W, requires_grad=False).view(1, -1)[None, None].to(device).long().repeat(N, C,
                                                                                                          H,
                                                                                                          1).view(
        -1)
    flat_weight = weight.view(-1)
    flat_img = img.contiguous().view(-1)

    # The corresponding positions in I1
    idxn = torch.arange(0, N, requires_grad=False).view(N, 1, 1, 1).to(device).long().repeat(1, C, H, W).view(
        -1)
    idxc = torch.arange(0, C, requires_grad=False).view(1, C, 1, 1).to(device).long().repeat(N, 1, H, W).view(
        -1)
    # ttype = flat_basex.type()
    idxx = flat_shiftx.long() + flat_basex
    idxy = flat_shifty.long() + flat_basey

    # recording the inside part the shifted
    mask = idxx.ge(0) & idxx.lt(H) & idxy.ge(0) & idxy.lt(W)

    # Mask off points out of boundaries
    ids = (idxn * C * H * W + idxc * H * W + idxx * W + idxy)
    ids_mask = torch.masked_select(ids, mask).clone().to(device)

    # Note here! accmulate fla must be true for proper bp
    img_warp = torch.zeros([N * C * H * W, ]).to(device)
    img_warp.put_(ids_mask, torch.masked_select(flat_img * flat_weight, mask), accumulate=True)

    one_warp = torch.zeros([N * C * H * W, ]).to(device)
    one_warp.put_(ids_mask, torch.masked_select(flat_weight, mask), accumulate=True)

    return img_warp.view(N, C, H, W), one_warp.view(N, C, H, W)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.PReLU(out_planes)
    )


def conv_woact(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
    )

def conv_wo_act(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
    )

def get_same_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.relu = nn.PReLU(planes)

        self.res_translate = None
        if not inplanes == planes or not stride == 1:
            self.res_translate = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if self.res_translate is not None:
            residual = self.res_translate(residual)
        out += residual

        return out

class IFBlock(nn.Module):
    def __init__(self, in_planes, scale=1, c=64, num_flows=3, mode='backward'):
        super(IFBlock, self).__init__()
        self.scale = scale
        self.conv0 = nn.Sequential(
            conv(in_planes, c, 3, 2, 1),
            conv(c, 2 * c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
            conv(2 * c, 2 * c),
        )
        if mode == 'backward':
            self.conv1 = nn.ConvTranspose2d(2 * c, 2 * num_flows * 2, 4, 2, 1)  # TODO WARNING: Notable change
        elif mode == 'forward':
            self.conv1 = nn.ConvTranspose2d(2 * c, 2 * num_flows * 3, 4, 2, 1)  # TODO WARNING: Notable change
        else:
            raise ValueError

    def forward(self, x):
        if self.scale != 1:
            x = F.interpolate(x, scale_factor=1. / self.scale, mode="bilinear",
                              align_corners=False)
            # print(x.size())

        x = self.conv0(x)
        x = self.convblock(x)
        flow = self.conv1(x)
        # print(flow.size())
        if self.scale != 1:
            flow = F.interpolate(flow, scale_factor=self.scale, mode="bilinear",
                                 align_corners=False)
            # print(flow.size())
        return flow


class FlowNetMulCatFusion(nn.Module):
    def __init__(self, num_flows=3, n_feats=32):
        super(FlowNetMulCatFusion, self).__init__()
        self.num_flows = num_flows
        self.block0 = IFBlock(in_planes=n_feats * 2 + 2 * num_flows, scale=8, c=192, num_flows=num_flows)
        self.conv0 = conv_wo_act(2 * 2 * num_flows, 2 * 2 * num_flows, 1, 1, 0)
        self.block1 = IFBlock(in_planes=2 * (2 + n_feats + 1) * num_flows, scale=4, c=128, num_flows=num_flows)
        self.conv1 = conv_wo_act(2 * 2 * num_flows, 2 * 2 * num_flows, 1, 1, 0)
        self.block2 = IFBlock(in_planes=2 * (2 + n_feats + 1) * num_flows, scale=2, c=96, num_flows=num_flows)
        self.conv2 = conv_wo_act(2 * 2 * num_flows, 2 * 2 * num_flows, 1, 1, 0)
        self.block3 = IFBlock(in_planes=2 * (2 + n_feats + 1) * num_flows, scale=1, c=48, num_flows=num_flows)
        self.conv3 = conv_wo_act(2 * 2 * num_flows, 2 * 2 * num_flows, 1, 1, 0)

    def _mul_encoding(self, flows, encodings):
        n, c, h, w = flows.shape
        assert encodings.shape == (n, int(c / 2), h, w), '{} != {}'.format(encodings.shape, (n, int(c / 2), h, w))
        flows = flows.reshape(n, int(c / 2), 2, h, w)
        encodings = encodings.reshape(n, int(c / 2), 1, h, w)
        flows *= encodings
        flows = flows.reshape(n, c, h, w)
        return flows

    def forward(self, x, encoding, return_velocity=False):
        x_t2b, x_b2t = torch.chunk(x, chunks=2, dim=1)  # (n, 32, h, w)
        encoding_ds = F.interpolate(encoding, scale_factor=0.5, mode='bilinear', align_corners=False,
                                    recompute_scale_factor=False)
        # print(x.size())
        flow0 = self.block0(torch.cat((x, encoding), dim=1))  # h/2,w/2
        # print(flow0.size())
        F1 = flow0
        F1 = self._mul_encoding(F1, encoding_ds)
        F1 = self.conv0(F1)
        F1_large = F.interpolate(F1, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        warped_t2b_imgs = multi_warp(x_t2b, F1_large[:, :2 * self.num_flows])
        warped_b2t_imgs = multi_warp(x_b2t, F1_large[:, 2 * self.num_flows:])
        warped_imgs = torch.cat((warped_t2b_imgs, warped_b2t_imgs), dim=1)  # (n, 2*num_flows*3, h, w)
        flow1 = self.block1(torch.cat((warped_imgs, encoding, F1_large), dim=1)) # h/2,w/2
        F2 = (flow0 + flow1)
        F2 = self._mul_encoding(F2, encoding_ds)
        F2 = self.conv1(F2)
        F2_large = F.interpolate(F2, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        warped_t2b_imgs = multi_warp(x_t2b, F2_large[:, :2 * self.num_flows])
        warped_b2t_imgs = multi_warp(x_b2t, F2_large[:, 2 * self.num_flows:])
        warped_imgs = torch.cat((warped_t2b_imgs, warped_b2t_imgs), dim=1)  # (n, 2*num_flows*3, h, w)
        flow2 = self.block2(torch.cat((warped_imgs, encoding, F2_large), dim=1))  ## h/2,w/2
        F3 = (flow0 + flow1 + flow2)
        F3 = self._mul_encoding(F3, encoding_ds)
        F3 = self.conv2(F3)
        F3_large = F.interpolate(F3, scale_factor=2.0, mode="bilinear", align_corners=False,
                                 recompute_scale_factor=False) * 2.0

        warped_t2b_imgs = multi_warp(x_t2b, F3_large[:, :2 * self.num_flows])
        warped_b2t_imgs = multi_warp(x_b2t, F3_large[:, 2 * self.num_flows:])
        warped_imgs = torch.cat((warped_t2b_imgs, warped_b2t_imgs), dim=1)  # (n, 2*num_flows*3, h, w)
        flow3 = self.block3(torch.cat((warped_imgs, encoding, F3_large), dim=1))   ## h/2,w/2
        F4 = (flow0 + flow1 + flow2 + flow3)
        F4 = self._mul_encoding(F4, encoding_ds)
        F4 = self.conv3(F4)

        if return_velocity:
            return F4, [F1, F2, F3, F4], flow0 + flow1 + flow2 + flow3
        return F4, [F1, F2, F3, F4]


class ConvDS(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(ConvDS, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SmallMaskNet(nn.Module):
    """A three-layer network for predicting mask"""
    def __init__(self, input, output):
        super(SmallMaskNet, self).__init__()
        self.conv1 = conv(input, 32, 5, padding=2)
        self.conv2 = conv(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, output, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class WarpedContextNet(nn.Module):
    def __init__(self, c=16, num_flows=3):
        super(WarpedContextNet, self).__init__()
        # self.num_flows = num_flows
        self.conv0_0 = ConvDS(3, c)
        self.conv0_1 = conv(2 * c, c, kernel_size=1, padding=0, stride=1)
        self.masknet0 = SmallMaskNet(c*2+4, 1)
        self.conv1_0 = ConvDS(c, c)
        self.conv1_1 = conv(2 * c, c, kernel_size=1, padding=0, stride=1)
        self.masknet1 = SmallMaskNet(c * 2 + 4, 1)
        self.conv2_0 = ConvDS(c, 2 * c)
        self.conv2_1 = conv(2 * (2 * c), 2 * c, kernel_size=1, padding=0, stride=1)
        self.masknet2 = SmallMaskNet(2*c * 2 + 4, 1)
        self.conv3_0 = ConvDS(2 * c, 4 * c)
        self.conv3_1 = conv(2 * (4 * c), 4 * c, kernel_size=1, padding=0, stride=1)
        self.masknet3 = SmallMaskNet(2 * c * 4 + 4, 1)
        self.conv4_0 = ConvDS(4 * c, 8 * c)
        self.conv4_1 = conv(2 * (8 * c), 8 * c, kernel_size=1, padding=0, stride=1)
        self.masknet4 = SmallMaskNet(2 * c * 8 + 4, 1)
        self.device = 'cuda'


    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        grid = grid.to(self.device)
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, padding_mode='border')
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        mask[mask < 0.999] = 0
        mask[mask > 0] = 1

        output = output * mask

        return output

    def forward(self, x, flow, encoding, is_b2t):  #encoding : b,3,1,h,w
        time_code = [encoding[:, i] for i in range(encoding.size()[1])]
        x = [self.conv0_0(xi) for xi in x]  #0,1,2
        flow = [F.interpolate(flowi, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5 for flowi in flow]  #f01,f10,f12,f21
        time_code = [F.interpolate(time_codei, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=False) for time_codei in time_code]
        time_coding1, time_coding2, warpmask = time_code[0], time_code[1], time_code[2]
        ft0_left, ft1_left = CFR_flow_t_align('cuda', flow[0], flow[1], time_coding1)
        ft1_right, ft2_right = CFR_flow_t_align('cuda', flow[2], flow[3], time_coding2)
        occ_left = self.masknet0(torch.cat([x[0], x[1], ft0_left, ft1_left], dim=1))
        occ_right = self.masknet0(torch.cat([x[1], x[2], ft1_right, ft2_right], dim=1))
        occ_0_left, occ_0_right = torch.sigmoid(occ_left), torch.sigmoid(occ_right)
        occ_1_left, occ_1_right = 1-occ_0_left, 1-occ_0_right
        Ft_left = (1 - time_coding1) * occ_0_left * self.warp(x[0], ft0_left) + time_coding1 * occ_1_left * self.warp(x[1], ft1_left)
        Ft_left = Ft_left / ((1 - time_coding1) * occ_0_left + time_coding1 * occ_1_left)
        Ft_right = (1 - time_coding2) * occ_0_right * self.warp(x[1], ft1_right) + time_coding2 * occ_1_right * self.warp(x[2], ft2_right)
        Ft_right = Ft_right / ((1 - time_coding2) * occ_0_right + time_coding2 * occ_1_right)
        f0 = self.conv0_1(torch.cat([Ft_left*warpmask, Ft_right*(1-warpmask)], dim=1))

        x = [self.conv1_0(xi) for xi in x]
        flow = [F.interpolate(flowi, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5 for flowi in flow]
        time_code = [F.interpolate(time_codei, scale_factor=0.5, mode='bilinear', align_corners=False,
                                   recompute_scale_factor=False) for time_codei in time_code]
        time_coding1, time_coding2, warpmask = time_code[0], time_code[1], time_code[2]
        ft0_left, ft1_left = CFR_flow_t_align('cuda', flow[0], flow[1], time_coding1)
        ft1_right, ft2_right = CFR_flow_t_align('cuda', flow[2], flow[3], time_coding2)
        occ_left = self.masknet1(torch.cat([x[0], x[1], ft0_left, ft1_left], dim=1))
        occ_right = self.masknet1(torch.cat([x[1], x[2], ft1_right, ft2_right], dim=1))
        occ_0_left, occ_0_right = torch.sigmoid(occ_left), torch.sigmoid(occ_right)
        occ_1_left, occ_1_right = 1 - occ_0_left, 1 - occ_0_right
        Ft_left = (1 - time_coding1) * occ_0_left * self.warp(x[0], ft0_left) + time_coding1 * occ_1_left * self.warp(
            x[1], ft1_left)
        Ft_left = Ft_left / ((1 - time_coding1) * occ_0_left + time_coding1 * occ_1_left)
        Ft_right = (1 - time_coding2) * occ_0_right * self.warp(x[1],
                                                                ft1_right) + time_coding2 * occ_1_right * self.warp(
            x[2], ft2_right)
        Ft_right = Ft_right / ((1 - time_coding2) * occ_0_right + time_coding2 * occ_1_right)
        f1 = self.conv1_1(torch.cat([Ft_left*warpmask, Ft_right*(1-warpmask)], dim=1))

        x = [self.conv2_0(xi) for xi in x]
        flow = [F.interpolate(flowi, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5 for flowi in flow]
        time_code = [F.interpolate(time_codei, scale_factor=0.5, mode='bilinear', align_corners=False,
                                   recompute_scale_factor=False) for time_codei in time_code]
        time_coding1, time_coding2, warpmask = time_code[0], time_code[1], time_code[2]
        ft0_left, ft1_left = CFR_flow_t_align('cuda', flow[0], flow[1], time_coding1)
        ft1_right, ft2_right = CFR_flow_t_align('cuda', flow[2], flow[3], time_coding2)
        occ_left = self.masknet2(torch.cat([x[0], x[1], ft0_left, ft1_left], dim=1))
        occ_right = self.masknet2(torch.cat([x[1], x[2], ft1_right, ft2_right], dim=1))
        occ_0_left, occ_0_right = torch.sigmoid(occ_left), torch.sigmoid(occ_right)
        occ_1_left, occ_1_right = 1 - occ_0_left, 1 - occ_0_right
        Ft_left = (1 - time_coding1) * occ_0_left * self.warp(x[0], ft0_left) + time_coding1 * occ_1_left * self.warp(
            x[1], ft1_left)
        Ft_left = Ft_left / ((1 - time_coding1) * occ_0_left + time_coding1 * occ_1_left)
        Ft_right = (1 - time_coding2) * occ_0_right * self.warp(x[1],
                                                                ft1_right) + time_coding2 * occ_1_right * self.warp(
            x[2], ft2_right)
        Ft_right = Ft_right / ((1 - time_coding2) * occ_0_right + time_coding2 * occ_1_right)
        f2 = self.conv2_1(torch.cat([Ft_left*warpmask, Ft_right*(1-warpmask)], dim=1))

        x = [self.conv3_0(xi) for xi in x]
        flow = [F.interpolate(flowi, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5 for flowi in flow]
        time_code = [F.interpolate(time_codei, scale_factor=0.5, mode='bilinear', align_corners=False,
                                   recompute_scale_factor=False) for time_codei in time_code]
        time_coding1, time_coding2, warpmask = time_code[0], time_code[1], time_code[2]
        ft0_left, ft1_left = CFR_flow_t_align('cuda', flow[0], flow[1], time_coding1)
        ft1_right, ft2_right = CFR_flow_t_align('cuda', flow[2], flow[3], time_coding2)
        occ_left = self.masknet3(torch.cat([x[0], x[1], ft0_left, ft1_left], dim=1))
        occ_right = self.masknet3(torch.cat([x[1], x[2], ft1_right, ft2_right], dim=1))
        occ_0_left, occ_0_right = torch.sigmoid(occ_left), torch.sigmoid(occ_right)
        occ_1_left, occ_1_right = 1 - occ_0_left, 1 - occ_0_right
        Ft_left = (1 - time_coding1) * occ_0_left * self.warp(x[0], ft0_left) + time_coding1 * occ_1_left * self.warp(
            x[1], ft1_left)
        Ft_left = Ft_left / ((1 - time_coding1) * occ_0_left + time_coding1 * occ_1_left)
        Ft_right = (1 - time_coding2) * occ_0_right * self.warp(x[1],
                                                                ft1_right) + time_coding2 * occ_1_right * self.warp(
            x[2], ft2_right)
        Ft_right = Ft_right / ((1 - time_coding2) * occ_0_right + time_coding2 * occ_1_right)
        f3 = self.conv3_1(torch.cat([Ft_left*warpmask, Ft_right*(1-warpmask)], dim=1))

        x = [self.conv4_0(xi) for xi in x]
        flow = [F.interpolate(flowi, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5 for flowi in flow]
        time_code = [F.interpolate(time_codei, scale_factor=0.5, mode='bilinear', align_corners=False,
                                   recompute_scale_factor=False) for time_codei in time_code]
        time_coding1, time_coding2, warpmask = time_code[0], time_code[1], time_code[2]
        ft0_left, ft1_left = CFR_flow_t_align('cuda', flow[0], flow[1], time_coding1)
        ft1_right, ft2_right = CFR_flow_t_align('cuda', flow[2], flow[3], time_coding2)
        occ_left = self.masknet4(torch.cat([x[0], x[1], ft0_left, ft1_left], dim=1))
        occ_right = self.masknet4(torch.cat([x[1], x[2], ft1_right, ft2_right], dim=1))
        occ_0_left, occ_0_right = torch.sigmoid(occ_left), torch.sigmoid(occ_right)
        occ_1_left, occ_1_right = 1 - occ_0_left, 1 - occ_0_right
        Ft_left = (1 - time_coding1) * occ_0_left * self.warp(x[0], ft0_left) + time_coding1 * occ_1_left * self.warp(
            x[1], ft1_left)
        Ft_left = Ft_left / ((1 - time_coding1) * occ_0_left + time_coding1 * occ_1_left)
        Ft_right = (1 - time_coding2) * occ_0_right * self.warp(x[1],
                                                                ft1_right) + time_coding2 * occ_1_right * self.warp(
            x[2], ft2_right)
        Ft_right = Ft_right / ((1 - time_coding2) * occ_0_right + time_coding2 * occ_1_right)
        f4 = self.conv4_1(torch.cat([Ft_left*warpmask, Ft_right*(1-warpmask)], dim=1))
        return [f0, f1, f2, f3, f4]


class IFEDNet(nn.Module):
    def __init__(self, c=16, num_flows=1):
        # self.num_flows = num_flows
        super(IFEDNet, self).__init__()
        self.conv0 = ConvDS(3 * 3, c)
        self.down0 = ConvDS(c*1, 2 * c)  # +c
        self.down1 = ConvDS(2 * c, 4 * c)  # +c
        self.down2 = ConvDS(4 * c, 8 * c)  # +2c
        self.down3 = ConvDS(8 * c, 16 * c)  # +4c
        self.up0 = deconv(16 * c, 8 * c)  # +8c
        self.up1 = deconv(16 * c, 4 * c)
        self.up2 = deconv(8 * c, 2 * c)
        self.up3 = deconv(4 * c, c)
        self.conv1 = deconv(c, 3, 4, 2, 1)
        # self.out = nn.Conv2d(c, 3, 3, 1, 1)
        self.masknet = SmallMaskNet(2 * 3 + 4, 1)
        self.device = 'cuda'
    # def gen_coding_time(self, x, reverse):
    #     b, c, h, w = x.size()
    #     time_coding1, _, _ = time_map(h, w, h, reverse)
    #     time_coding1, _, _ = time_coding1.permute(2, 0, 1) #, time_coding2.permute(2, 0, 1), warpmask.permute(2, 0, 1)
    #     time_coding1, _, _ = time_coding1.repeat(b, 1, 1, 1) #, time_coding2.repeat(b, 1, 1, 1), warpmask.repeat(b, 1, 1, 1)
    #
    #     return time_coding1 #, time_coding2, warpmask

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        grid = grid.to(self.device)
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, padding_mode='border')
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        mask[mask < 0.999] = 0
        mask[mask > 0] = 1

        output = output * mask

        return output

    def forward(self, x0, x2, flow02, flow20, encoding, is_b2t):  #rs_cube channel: c,c,2c,4c,8c   rs_cube,
        time_code = [encoding[:, i] for i in range(encoding.size()[1])]
        time_coding = time_code[0]
        ft0, ft2 = CFR_flow_t_align('cuda', flow02, flow20, time_coding)
        occ = self.masknet(torch.cat([x0, x2, ft0, ft2], dim=1))
        occ_0 = torch.sigmoid(occ)
        occ_1 = 1 - occ_0
        warped_img = (1 - time_coding) * occ_0 * self.warp(x0, ft0) + time_coding * occ_1 * self.warp(x2, ft2)

        d0 = self.conv0(torch.cat((warped_img, x0, x2), dim=1))
        d0 = self.down0(d0)   #torch.cat([d0, rs_cube[0]], dim=1)
        d1 = self.down1(d0)    #torch.cat((d0, rs_cube[1]), dim=1)
        d2 = self.down2(d1)    #torch.cat((d1, rs_cube[2]), dim=1)
        d3 = self.down3(d2)     #torch.cat((d2, rs_cube[3]), dim=1)
        out = self.up0(d3)   #torch.cat((d3, rs_cube[4]), dim=1)
        out = self.up1(torch.cat((out, d2), dim=1))
        out = self.up2(torch.cat((out, d1), dim=1))
        out = self.up3(torch.cat((out, d0), dim=1))
        out = self.conv1(out)
        # out = self.out(out)
        res = torch.sigmoid(out) * 2 - 1
        pred = warped_img + res
        pred = torch.clamp(pred, 0, 1)

        return pred



class RSG(nn.Module):
    def __init__(self, num_frames, n_feats, load_flow_net, flow_pretrain_fn):
        super(RSG, self).__init__()
        # self.warped_context_net = WarpedContextNet(c=n_feats, num_flows=num_frames)
        self.ife_net = IFEDNet(c=n_feats, num_flows=num_frames)
        self.pwcflow = flow_pwc.Flow_PWC(load_pretrain=load_flow_net, pretrain_fn=flow_pretrain_fn, device='cuda')

    def forward(self, x, encoding, is_b2t):
        #x:b,3,c,h,w        encoding: b,2,3,1,h,w
        # flow01 = self.pwcflow(x[:, 0], x[:, 1])
        # flow10 = self.pwcflow(x[:, 1], x[:, 0])
        #
        # flow12 = self.pwcflow(x[:, 1], x[:, 2])
        # flow21 = self.pwcflow(x[:, 2], x[:, 1])
        # print(encoding.size())
        # rs_cube = self.warped_context_net([x[:, 0], x[:, 1], x[:, 2]], [flow01, flow10, flow12, flow21], encoding[:, 0], is_b2t)

        flow02 = self.pwcflow(x[:, 0], x[:, 2])
        flow20 = self.pwcflow(x[:, 2], x[:, 0])

        out = self.ife_net(x[:, 0], x[:, 2], flow02, flow20, encoding[:, 1], is_b2t)  #rs_cube,

        return out



def cost_profile(model, H, W, seq_length, ext_frames):
    x = torch.randn(1, 6 * seq_length, H, W).cuda()
    x = pad(x)
    encodings = torch.randn(1, 2 * ext_frames, H, W).cuda()
    encodings = pad(encodings)
    flops, params = profile(model, inputs=(x, encodings), verbose=False)

    return flops, params

#
# if __name__ == '__main__':
#     from para import Parameter
#
#     args = Parameter().args
#     inputs = torch.randn(4, 18, 256, 256).cuda()
#     encodings = torch.randn(4, 6, 256, 256).cuda()
#     model = Model(args).cuda()
#     outputs, flows = model(inputs, encodings)
#     print(outputs.shape)
#     for flow in flows:
#         print(flow.shape)

