import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import warnings
from thop import profile
try:
    from .warplayer import multi_warp
except:
    from model.warplayer import multi_warp
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from models import flow_pwc

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
    def __init__(self, num_flows=3):
        super(FlowNetMulCatFusion, self).__init__()
        self.num_flows = num_flows
        self.block0 = IFBlock(in_planes=6 + 2 * num_flows, scale=8, c=192, num_flows=num_flows)
        self.conv0 = conv_wo_act(2 * 2 * num_flows, 2 * 2 * num_flows, 1, 1, 0)
        self.block1 = IFBlock(in_planes=2 * (2 + 3 + 1) * num_flows, scale=4, c=128, num_flows=num_flows)
        self.conv1 = conv_wo_act(2 * 2 * num_flows, 2 * 2 * num_flows, 1, 1, 0)
        self.block2 = IFBlock(in_planes=2 * (2 + 3 + 1) * num_flows, scale=2, c=96, num_flows=num_flows)
        self.conv2 = conv_wo_act(2 * 2 * num_flows, 2 * 2 * num_flows, 1, 1, 0)
        self.block3 = IFBlock(in_planes=2 * (2 + 3 + 1) * num_flows, scale=1, c=48, num_flows=num_flows)
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
        x_t2b, x_b2t = torch.chunk(x, chunks=2, dim=1)  # (n,  3, h, w)
        encoding_ds = F.interpolate(encoding, scale_factor=0.5, mode='bilinear', align_corners=False,
                                    recompute_scale_factor=False)

        flow0 = self.block0(torch.cat((x, encoding), dim=1))  # h/2,w/2
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


class WarpedContextNet(nn.Module):
    def __init__(self, c=16, num_flows=3):
        super(WarpedContextNet, self).__init__()
        self.num_flows = num_flows
        self.conv0_0 = ConvDS(3, c)
        self.conv1_0 = ConvDS(c, c)
        self.conv1_1 = conv(num_flows * c, c, kernel_size=1, padding=0, stride=1)
        self.conv2_0 = ConvDS(c, 2 * c)
        self.conv2_1 = conv(num_flows * (2 * c), 2 * c, kernel_size=1, padding=0, stride=1)
        self.conv3_0 = ConvDS(2 * c, 4 * c)
        self.conv3_1 = conv(num_flows * (4 * c), 4 * c, kernel_size=1, padding=0, stride=1)
        self.conv4_0 = ConvDS(4 * c, 8 * c)
        self.conv4_1 = conv(num_flows * (8 * c), 8 * c, kernel_size=1, padding=0, stride=1)

    def forward(self, x, flow):
        x = self.conv0_0(x)
        x = self.conv1_0(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f1 = multi_warp(x, flow)
        f1 = self.conv1_1(f1)

        x = self.conv2_0(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f2 = multi_warp(x, flow)
        f2 = self.conv2_1(f2)

        x = self.conv3_0(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f3 = multi_warp(x, flow)
        f3 = self.conv3_1(f3)

        x = self.conv4_0(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f4 = multi_warp(x, flow)
        f4 = self.conv4_1(f4)
        return [f1, f2, f3, f4]


class IFEDNet(nn.Module):
    def __init__(self, c=16, num_flows=3):
        self.num_flows = num_flows
        super(IFEDNet, self).__init__()
        self.conv0 = ConvDS(2 * (3 + 2) * num_flows, c)
        self.down0 = ConvDS(c, 2 * c)
        self.down1 = ConvDS(4 * c, 4 * c)  # +2c
        self.down2 = ConvDS(8 * c, 8 * c)  # +4c
        self.down3 = ConvDS(16 * c, 16 * c)  # +8c
        self.up0 = deconv(32 * c, 8 * c)  # +16c
        self.up1 = deconv(16 * c, 4 * c)
        self.up2 = deconv(8 * c, 2 * c)
        self.up3 = deconv(4 * c, c)
        self.conv1 = deconv(c, 4 * num_flows, 4, 2, 1)

    def forward(self, img_t2b, img_b2t, flow_t2b, flow_b2t, c_t2b, c_b2t):
        warped_t2b_imgs = multi_warp(img_t2b, flow_t2b)  # (n, num_flows*3, h, w)
        warped_b2t_imgs = multi_warp(img_b2t, flow_b2t)  # (n, num_flows*3, h, w)

        d0 = self.conv0(torch.cat((warped_t2b_imgs, warped_b2t_imgs, flow_t2b, flow_b2t), dim=1))
        d0 = self.down0(d0)
        d1 = self.down1(torch.cat((d0, c_t2b[0], c_b2t[0]), dim=1))
        d2 = self.down2(torch.cat((d1, c_t2b[1], c_b2t[1]), dim=1))
        d3 = self.down3(torch.cat((d2, c_t2b[2], c_b2t[2]), dim=1))
        out = self.up0(torch.cat((d3, c_t2b[3], c_b2t[3]), dim=1))
        out = self.up1(torch.cat((out, d2), dim=1))
        out = self.up2(torch.cat((out, d1), dim=1))
        out = self.up3(torch.cat((out, d0), dim=1))
        out = self.conv1(out)

        res = torch.sigmoid(out[:, :3 * self.num_flows]) * 2 - 1
        mask = torch.sigmoid(out[:, 3 * self.num_flows:])  # (n, 3, h, w)
        n, c, h, w = warped_t2b_imgs.shape
        warped_t2b_imgs = warped_t2b_imgs.reshape(n, self.num_flows, 3, h, w)
        warped_b2t_imgs = warped_b2t_imgs.reshape(n, self.num_flows, 3, h, w)
        mask = mask.reshape(n, self.num_flows, 1, h, w)
        warped_imgs = mask * warped_t2b_imgs + (1. - mask) * warped_b2t_imgs
        warped_imgs = warped_imgs.reshape(n, self.num_flows * 3, h, w)
        pred = warped_imgs + res
        pred = torch.clamp(pred, 0, 1)

        return pred



class RSG(nn.Module):
    def __init__(self, num_frames, n_feats, load_flow_net, flow_pretrain_fn):
        super(RSG, self).__init__()
        self.flow_net = FlowNetMulCatFusion(num_flows=num_frames)
        self.warped_context_net = WarpedContextNet(c=n_feats, num_flows=num_frames)
        self.ife_net = IFEDNet(c=n_feats, num_flows=num_frames)

    def forward(self, x, encoding, return_velocity=False):
        x_t2b, x_b2t = x[:, 0], x[:, 1]
        x = torch.cat((x_t2b, x_b2t), dim=1)
        if return_velocity:
            flow, flows, velocity = self.flow_net(x, encoding, return_velocity)
        else:
            flow, flows = self.flow_net(x, encoding)  ## h/2,w/2
        flow_t2b, flow_b2t = torch.chunk(flow, chunks=2, dim=1)
        c_t2b = self.warped_context_net(x_t2b, flow_t2b)
        c_b2t = self.warped_context_net(x_b2t, flow_b2t)
        flow_t2b = F.interpolate(flow_t2b, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0  # h,w
        flow_b2t = F.interpolate(flow_b2t, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        out = self.ife_net(x_t2b, x_b2t, flow_t2b, flow_b2t, c_t2b, c_b2t)

        if return_velocity:
            return out, flows, velocity
        return out, flows



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

