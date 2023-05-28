from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.optim import Adam, AdamW, Adamax

from models.select_network import define_G
from models.model_plain import ModelPlain
from models.flow_pwc import Flow_PWC as net
from models.network_rsgenerator import CFR_flow_t_align
from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip
from torch.autograd import Variable

try:
    from .loss import *
except:
    from loss import *

class ModelSRSCRSG(ModelPlain):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(ModelSRSCRSG, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()
        self.fix_iter = self.opt_train.get('fix_iter', 0)
        self.fix_keys = self.opt_train.get('fix_keys', [])
        self.fix_unflagged = True
        self.net_rsg = net(load_pretrain=True, pretrain_fn=self.opt['path']['pretrained_rsg'], device='cuda')
        self.net_rsg = self.model_to_device(self.net_rsg)
        # print('Loading model for RS Generator [{:s}] ...'.format(self.opt['path']['pretrained_rsg']))
        # self.load_network(self.opt['path']['pretrained_rsg'], self.net_rsg, strict=True, param_key='params')

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------

    def define_loss(self):
        # G_lossfn_type = self.opt_train['G_lossfn_type']
        ratios, losses = loss_parse(self.opt_train['G_lossfn_type'])
        self.losses_name = losses
        self.ratios = ratios
        self.losses = []
        for loss in losses:
            loss_fn = eval('{}(self.opt_train)'.format(loss))
            self.losses.append(loss_fn)

        self.G_lossfn_weight = self.ratios   #self.opt_train['G_lossfn_weight']


    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        self.fix_keys = self.opt_train.get('fix_keys', [])
        if self.opt_train.get('fix_iter', 0) and len(self.fix_keys) > 0:
            fix_lr_mul = self.opt_train['fix_lr_mul']
            print(f'Multiple the learning rate for keys: {self.fix_keys} with {fix_lr_mul}.')
            if fix_lr_mul == 1:
                G_optim_params = self.netG.parameters()
            else:  # separate flow params and normal params for different lr
                normal_params = []
                flow_params = []
                for name, param in self.netG.named_parameters():
                    if any([key in name for key in self.fix_keys]):
                        flow_params.append(param)
                    else:
                        normal_params.append(param)
                G_optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': self.opt_train['G_optimizer_lr']
                    },
                    {
                        'params': flow_params,
                        'lr': self.opt_train['G_optimizer_lr'] * fix_lr_mul
                    },
                ]
        else:
            G_optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    G_optim_params.append(v)
                else:
                    print('Params [{:s}] will not optimize.'.format(k))
        # self.params = G_optim_params
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'])
        elif self.opt_train['G_optimizer_type'] == 'adamw':
            self.G_optimizer = AdamW(G_optim_params, lr=self.opt_train['G_optimizer_lr'])
        elif self.opt_train['G_optimizer_type'] == 'adamax':
            self.G_optimizer = Adamax(G_optim_params, lr=self.opt_train['G_optimizer_lr'])
        else:
            raise NotImplementedError

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        if self.fix_iter:
            if self.fix_unflagged and current_step < self.fix_iter:
                print(f'Fix keys: {self.fix_keys} for the first {self.fix_iter} iters.')
                self.fix_unflagged = False
                for name, param in self.netG.named_parameters():
                    if any([key in name for key in self.fix_keys]):
                        param.requires_grad_(False)
            elif current_step == self.fix_iter:
                print(f'Train all the parameters from {self.fix_iter} iters.')
                self.netG.requires_grad_(True)

        super(ModelSRSCRSG, self).optimize_parameters(current_step)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt_train['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                            self.opt_train['G_scheduler_milestones'],
                                                            self.opt_train['G_scheduler_gamma']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingWarmRestarts':
            self.schedulers.append(lr_scheduler.CosineAnnealingWarmRestarts(self.G_optimizer,
                                                            self.opt_train['G_scheduler_periods'],
                                                            self.opt_train['G_scheduler_restart_weights'],
                                                            self.opt_train['G_scheduler_eta_min']
                                                            ))
        else:
            raise NotImplementedError

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        # self.L = data['L'].to(self.device)
        # if need_H:
        #     self.H = data['H'].to(self.device)
        self.L, self.H, self.H_flows, self.dis_encodings, self.time_rsc, self.all_time_rsc, self.out_path, self.input_path = data   #rs_imgs, gs_imgs, fl_imgs, prior_imgs, time_rsc, out_paths, input_path
        self.L = self.L.to(self.device)
        self.H = self.H.to(self.device)
        # self.H_flows = self.H_flows.to(self.device)
        self.dis_encodings = self.dis_encodings.to(self.device)
        self.time_rsc = self.time_rsc.to(self.device)
        # print("input:", self.input_path, 'output:', self.out_path)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def pad(self, img, ratio=32):
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
        elif len(img.shape) == 6:
            b, n1, n2, c, h, w = img.shape
            img_list = []
            for i in range(n1):
                img1 = img[:, i].reshape(b * n2, c, h, w)
                ph = ((h - 1) // ratio + 1) * ratio
                pw = ((w - 1) // ratio + 1) * ratio
                padding = (0, pw - w, 0, ph - h)
                img1 = F.pad(img1, padding, mode='replicate')
                img1 = img1.reshape(b, n2, c, ph, pw)
                img_list.append(img1)
            img = torch.stack(img_list, dim=1)
            return img

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

    def netG_forward(self, is_train=True):

        b, n, c, h, w = self.L.shape  # (8, 3, 3, 256, 256)
        ori_h, ori_w = h, w
        if ori_h % 32 != 0 or ori_w % 32 != 0:
            self.L = self.pad(self.L)
            self.dis_encodings = self.pad(self.dis_encodings)
            self.time_rsc = self.pad(self.time_rsc)
            if not is_train:
                self.all_time_rsc = self.pad(self.all_time_rsc)
            # print(self.dis_encodings.size())
        # b, n, c, h, w = self.L.shape
        # self.L = self.L.reshape(b, n * c, h, w)  # (8, 18, 256, 256)

        # self.dis_encodings.shape  # (b, 2, 3, 1, 256, 256)
        # list = []
        # for idx in range(self.time_rsc.size()[1]):    # b,6,1,h,w
        #     if idx != 1 and idx != 4:
        #         list.append(self.time_rsc[:, idx])
        # time_rsc = torch.stack(list, dim=1)

        if is_train:
            b, n, c, h, w = self.time_rsc.shape  # (8, 2*ext_frames, 1, 256, 256)
            time_rsc = self.time_rsc.reshape(b, n * c, h, w)
            self.E, self.flows = self.netG(self.L, time_rsc)   #b, num_frames*3, h, w   -----   b, num_frames*2*2, h, w
            b, c, h, w = self.E.shape
            self.E = self.E.reshape(b, c // 3, 3, h, w)
            assert c // 3 == 3, c
        else:
            assert self.all_time_rsc.size()[1]//2 == 9, self.all_time_rsc.size()[1]
            E_list = []
            for idx in range(1, 8):  #self.time_rsc  b,6(t2b:0,index,8; b2t:0,index,8),1,h,w
                self.time_rsc[:, 1] = self.all_time_rsc[:, idx]
                self.time_rsc[:, 4] = self.all_time_rsc[:, idx+9]
                b, n, c, h, w = self.time_rsc.shape  # (8, 2*ext_frames, 1, 256, 256)
                time_rsc = self.time_rsc.reshape(b, n * c, h, w)
                self.E, self.flows = self.netG(self.L, time_rsc)  # b, num_frames*3, h, w   -----   b, num_frames*2*2, h, w
                b, c, h, w = self.E.shape
                self.E = self.E.reshape(b, c // 3, 3, h, w)
                assert c // 3 == 3, c
                if idx == 1:
                    E_list.append(self.E[:, 0])
                    E_list.append(self.E[:, 1])
                elif idx == 7:
                    E_list.append(self.E[:, 1])
                    E_list.append(self.E[:, 2])
                else:
                    E_list.append(self.E[:, 1])
            assert len(E_list) == 9, len(E_list)
            self.E = torch.stack(E_list, dim=1)

        if is_train:
            self.dis_encodings1, self.dis_encodings2 = torch.chunk(self.dis_encodings, chunks=2, dim=1)  #b,   whether reverse
            time_coding1 = self.dis_encodings1[:, 1, 0]
            time_coding2 = self.dis_encodings2[:, 1, 0]
            mid_time_coding1_up, mid_time_coding1_down = self.dis_encodings1[:, 0, 0], self.dis_encodings1[:, 0, 1]
            mid_time_coding2_up, mid_time_coding2_down = self.dis_encodings2[:, 0, 0], self.dis_encodings2[:, 0, 1]
            mid_mask1 = self.dis_encodings1[:, 0, 2]   #t2b
            mid_mask2 = self.dis_encodings2[:, 0, 2]   #b2t

            self.flow02 = self.net_rsg(self.E[:, 0], self.E[:, 2])
            self.flow20 = self.net_rsg(self.E[:, 2], self.E[:, 0])
            self.flow01 = self.net_rsg(self.E[:, 0], self.E[:, 1])
            self.flow10 = self.net_rsg(self.E[:, 1], self.E[:, 0])
            self.flow12 = self.net_rsg(self.E[:, 1], self.E[:, 2])
            self.flow21 = self.net_rsg(self.E[:, 2], self.E[:, 1])

            # whole
            ft0, ft2 = CFR_flow_t_align('cuda', self.flow02, self.flow20, time_coding1)
            self.L_t2b = (1 - time_coding1) * self.warp(self.E[:, 0], ft0) + time_coding1 * self.warp(self.E[:, 2], ft2)  #* occ_0   * occ_1
            ft0, ft2 = CFR_flow_t_align('cuda', self.flow02, self.flow20, time_coding2)
            self.L_b2t = (1 - time_coding2) * self.warp(self.E[:, 0], ft0) + time_coding2 * self.warp(self.E[:, 2], ft2)  #* occ_0   * occ_1

            # 0-1, 1-2
            ft0_f, ft1_f = CFR_flow_t_align('cuda', self.flow01, self.flow10, mid_time_coding1_up)
            warped_img_f = (1 - mid_time_coding1_up) * self.warp(self.E[:, 0], ft0_f) + mid_time_coding1_up * self.warp(self.E[:, 1], ft1_f)
            warped_img_f = warped_img_f * mid_mask1
            ft1_b, ft2_b = CFR_flow_t_align('cuda', self.flow12, self.flow21, mid_time_coding1_down)
            warped_img_b = (1 - mid_time_coding1_down) * self.warp(self.E[:, 1], ft1_b) + mid_time_coding1_down * self.warp(self.E[:, 2], ft2_b)
            warped_img_b = warped_img_b * (1 - mid_mask1)
            self.L_t2b_mid = warped_img_b + warped_img_f

            ft0_f, ft1_f = CFR_flow_t_align('cuda', self.flow01, self.flow10, mid_time_coding2_up)
            warped_img_f = (1 - mid_time_coding2_up) * self.warp(self.E[:, 0], ft0_f) + mid_time_coding2_up * self.warp(self.E[:, 1], ft1_f)
            warped_img_f = warped_img_f * mid_mask2
            ft1_b, ft2_b = CFR_flow_t_align('cuda', self.flow12, self.flow21, mid_time_coding2_down)
            warped_img_b = (1 - mid_time_coding2_down) * self.warp(self.E[:, 1], ft1_b) + mid_time_coding2_down * self.warp(self.E[:, 2], ft2_b)
            warped_img_b = warped_img_b * (1 - mid_mask2)
            self.L_b2t_mid = warped_img_b + warped_img_f

        if ori_h % 32 != 0 or ori_w % 32 != 0:
            if is_train:
                self.L_t2b = self.L_t2b[:, :, :ori_h, :ori_w]
                self.L_b2t = self.L_b2t[:, :, :ori_h, :ori_w]
                self.L_t2b_mid = self.L_t2b_mid[:, :, :ori_h, :ori_w]
                self.L_b2t_mid = self.L_b2t_mid[:, :, :ori_h, :ori_w]
            self.L = self.L[:, :, :, :ori_h, :ori_w]
            self.E = self.E[:, :, :, :ori_h, :ori_w]

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        # torch.autograd.set_detect_anomaly(True)
        self.netG_forward()

        # b, num_gs, c, h, w = self.H.shape  # (b, 3, 3, 256, 256)
        # self.H = self.H.reshape(b, num_gs * c, h, w).detach()

        # b, c, h, w = self.H.shape
        fb, fc, fh, fw = self.flows[0].shape
        assert fc == 12, fc
        # print([flow.size() for flow in self.flows[0]])
        losses = {}
        loss_all = None
        for i in range(len(self.losses)):
            if self.losses_name[i].lower().startswith('epe'):
                loss_sub = self.losses[i](self.flows[0], self.H_flows, 1)
                for flow in self.flows[1:]:
                    loss_sub += self.losses[i](flow, self.H_flows, 1)
                loss_sub = self.ratios[i] * loss_sub.mean()
            elif self.losses_name[i].lower().startswith('census'):
                loss_sub = self.ratios[i] * [self.losses[i](self.E.reshape(b * int(c // 3), 3, h, w), self.H.reshape(b * int(c // 3), 3, h, w)).mean()]
            elif self.losses_name[i].lower().startswith('variation'):
                loss_sub = self.losses[i](self.flows[0].reshape(fb * int(fc // 2), 2, fh, fw), mean=True)
                for flow in self.flows[1:]:
                    loss_sub += self.losses[i](flow.reshape(fb * int(fc // 2), 2, fh, fw), mean=True)
                #for flow in self.rs_flow:
                    #loss_sub += self.losses[i](flow, mean=True)
                loss_sub = self.ratios[i] * loss_sub
            # elif self.losses_name[i].lower().startswith('Charbonnier'):
            #     loss_sub = self.ratios[i] * (self.losses[i](self.L_t2b, self.L[:, 0]) + self.losses[i](self.L_b2t, self.L[:, 1]))  #self.ratios[i] * (self.losses[i](self.E, self.L) +
            else:
                loss_sub = self.ratios[i] * (self.losses[i](self.L_t2b, self.L[:, 0]) + self.losses[i](self.L_t2b_mid, self.L[:, 0]) + self.losses[i](self.L_b2t, self.L[:, 1]) + self.losses[i](self.L_b2t_mid, self.L[:, 1]))
            losses[self.losses_name[i]] = loss_sub
            self.log_dict[self.losses_name[i]] = loss_sub.item()
            if loss_all == None:
                loss_all = loss_sub
            else:
                loss_all += loss_sub
        G_loss = loss_all
        # with torch.autograd.detect_anomaly():
        G_loss.backward()
        # for name, param in self.netG.named_parameters():
        #     if param.grad.isnan():
        #         print(name)  #module.net_scale.conv1.weight
            # break
        # print(self.netG.module.net_scale.conv1.weight, self.netG.module.net_scale.conv1.weight.grad)

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward(False)
        # self.H = torch.chunk(self.H, chunks=self.H.size()[1]//3, dim=1)
        # b,c,h,w = self.E.shape
        # self.E = self.E.reshape(b, c//3, 3, h, w)
        self.netG.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        # out_dict['L'] = self.L.detach()[0].float().cpu()
        # out_dict['L_t2b'] = self.L_t2b.detach()[0].float().cpu()
        # out_dict['L_b2t'] = self.L_b2t.detach()[0].float().cpu()
        # out_dict['E0'] = self.E[:, 0].detach()[0].float().cpu()
        # out_dict['E1'] = self.E[:, 1].detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        # out_dict['L0'] = self.L[:, 0].detach()[0].float().cpu()
        # out_dict['L1'] = self.L[:, 1].detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
