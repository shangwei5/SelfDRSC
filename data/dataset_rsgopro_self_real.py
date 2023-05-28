"""
RS-GOPRO dataset
Synthesized rolling shutter distortion using high-fps video (GOPRO-Large)
"""

import os
import random
from os.path import join
from time import time

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data.distortion_prior import distortion_map, time_map
from data.utils import normalize, flow_to_image
# try:
#     from .utils import normalize, flow_to_image
#     from .distortion_prior import distortion_map
# except:
#     from utils import normalize, flow_to_image
#     from distortion_prior import distortion_map
from models.warplayer import warp


class RSGOPRO(Dataset):
    """ Dataset class for RS-GOPRO"""

    def __init__(self, path, future_frames=1, past_frames=1, ext_frames=17, crop_size=256, centralize=False,
                 normalize=True, flow=False, is_test=False, inference=False):
        """
        Initialize dataset class

        Paramters:
            path: where GOPRO-RS locates
            future_frames: number of frames in the future
            past_frames: number of frames in the past
            ext_frames: number of extracted ground truth frames for single rs frame, [1 | 3 | 5| 9]
            crop_size: patch size of random cropping
            centralize: subtract half value range
            normalize: divide value range
        """
        super(RSGOPRO, self).__init__()
        self.H = 640
        self.W = 640
        # self.H = 540
        # self.W = 960
        self.flow = flow
        # assert ext_frames in [1, 3, 5, 9], "ext_frames should be [1 | 3 | 5 | 9]"
        self.ext_frames = ext_frames
        self.time_encoding = self._gen_time_coding(self.H, self.W, self.ext_frames)
        assert self.time_encoding.shape == (2 * self.ext_frames, 3, self.H, self.W, 1), self.time_encoding.shape
        self.dis_encoding = self._gen_dis_encoding(self.H, self.W, self.ext_frames)
        assert self.dis_encoding.shape == (2 * self.ext_frames, self.H, self.W, 1), self.dis_encoding.shape
        self.normalize = normalize
        self.centralize = centralize
        self.num_ff = future_frames
        self.num_pf = past_frames
        self.is_test = is_test
        self._samples = self._generate_samples(path)
        if crop_size is not None:
            self.crop_h, self.crop_w = crop_size, crop_size
        else:
            self.crop_h, self.crop_w = self.H, self.W
        self.inference = inference

    def _generate_samples(self, path):
        samples = []
        # if self.is_test:
        #     seqs = [seq for seq in sorted(os.listdir(path))[0:1] if not seq.endswith('avi')]
        # else:
        seqs = [seq for seq in sorted(os.listdir(path)) if not seq.endswith('avi')]
        for seq in seqs:
            if seq == 's0' or seq == 's1':   # or seq == 'r1':
                print("Sequence {} is not used.".format(seq))
                continue
            seq_path = join(path, seq)
            seq_num = len(os.listdir(seq_path)) // 2   #include t2b and b2t
            for i in range(self.num_pf, seq_num - self.num_ff):
                sample = dict()
                sample['RS'] = {}
                sample['RS']['t2b'] = [join(seq_path, '{:01d}_t2b.png'.format(j)) for j in
                                       range(i - self.num_pf, i + self.num_ff + 1)]
                sample['RS']['b2t'] = [join(seq_path, '{:01d}_b2t.png'.format(j)) for j in
                                       range(i - self.num_pf, i + self.num_ff + 1)]
                # sample['RS']['b2t'] = [join(seq_path, 'RS', '{:08d}_rs_t2b.png'.format(j)) for j in
                #                        range(i - self.num_pf, i + self.num_ff + 1)]

                if self.ext_frames == 1:
                    gs_indices = [4]
                elif self.ext_frames == 3:
                    gs_indices = [0, 4, 8]
                elif self.ext_frames == 5:
                    gs_indices = [0, 2, 4, 6, 8]
                elif self.ext_frames == 9:
                    gs_indices = list(range(9))
                else:
                    gs_indices = list(range(self.ext_frames))
                self.gs_indices = gs_indices
                samples.append(sample)
        return samples

    def _gen_dis_encoding(self, h, w, ext_frames):
        dis_encoding = []
        if ext_frames == 1:
            ref_rows = [(h - 1) / 2, ]
        else:
            ref_rows = np.linspace(0, h - 1, ext_frames)
        for ref_row in ref_rows:
            dis_encoding.append(distortion_map(h, w, ref_row)[..., np.newaxis])
        for ref_row in ref_rows:
            dis_encoding.append(distortion_map(h, w, ref_row, reverse=True)[..., np.newaxis])
        dis_encoding = np.stack(dis_encoding, axis=0)  # (2*ext_frames, h, w, 1)
        return dis_encoding

    def _gen_time_coding(self, h, w, ext_frames):
        time_coding = []
        if ext_frames == 1:
            ref_rows = [(h - 1) / 2, ]
        else:
            ref_rows = np.linspace(0, h - 1, ext_frames)
        # print(ref_rows.shape)
        for ref_row in ref_rows:
            time_coding.append(time_map(h, w, ref_row))
        for ref_row in ref_rows:
            time_coding.append(time_map(h, w, ref_row, reverse=True))
        # print(time_coding[0].shape)
        time_coding = np.stack(time_coding, axis=0)  # (2*ext_frames, 3, h, w, 1)
        # print(time_coding.shape)
        return time_coding

    def __len__(self):
        # if self.inference:
        #     return len(self._samples) * 2
        return len(self._samples)

    def __getitem__(self, idx):
        index = random.choice(self.gs_indices[1:-1])

        # scane_line = np.linspace(0, self.H - 1, self.ext_frames)[index] / (self.H - 1)
        rs_imgs, prior_imgs, time_rsc, all_time_rsc, out_paths = self._load_sample(self._samples[idx], index)
        assert rs_imgs.shape == (2*(self.num_ff + 1 + self.num_pf), 3, self.crop_h, self.crop_w), rs_imgs.shape  # t2b & b2t
        # if self.inference:
        #     assert gs_imgs.shape == (self.ext_frames, 3, self.crop_h, self.crop_w), gs_imgs.shape   #0,index,8   *(self.num_ff + 1 + self.num_pf)
        # else:
        #     assert gs_imgs.shape == (3, 3, self.crop_h, self.crop_w), gs_imgs.shape
        # assert fl_imgs.shape == (2 * self.ext_frames, 2, self.crop_h, self.crop_w), fl_imgs.shape    # * (self.num_ff + 1 + self.num_pf)
        assert prior_imgs.shape == (4, 3, 1, self.crop_h, self.crop_w), prior_imgs.shape   #(1/h)*index, 1, (timecode1,timecode2, warpmask)
        assert time_rsc.shape == (6, 1, self.crop_h, self.crop_w), time_rsc.shape
        assert all_time_rsc.shape == (2 * self.ext_frames, 1, self.crop_h, self.crop_w), all_time_rsc.shape
        rs_imgs = normalize(rs_imgs, normalize=self.normalize, centralize=self.centralize)
        # gs_imgs = normalize(gs_imgs, normalize=self.normalize, centralize=self.centralize)
        return rs_imgs, rs_imgs, rs_imgs, prior_imgs, time_rsc, all_time_rsc, out_paths, out_paths

    def _load_sample(self, sample, index):
        top = random.randint(0, self.H - self.crop_h)
        left = random.randint(0, self.W - self.crop_w)
        rs_imgs, gs_imgs, fl_imgs, prior_imgs, time_rsc, all_time_rsc, out_paths, input_paths = [], [], [], [], [], [], [], []
        for rs_img_path in sample['RS']['t2b']:
            img = self._data_augmentation(cv2.imread(rs_img_path), top, left)
            rs_imgs.append(img)
            out_paths.append(rs_img_path)

        for rs_img_path in sample['RS']['b2t']:
            img = self._data_augmentation(cv2.imread(rs_img_path), top, left)  # , flip=True
            rs_imgs.append(img)
            out_paths.append(rs_img_path)

        for i in range(self.time_encoding.shape[0]):
            if i % self.ext_frames == index or i % self.ext_frames == self.ext_frames-1:
                # if is_b2t:
                #     if i >= self.ext_frames:
                img = self._data_augmentation(self.time_encoding[i].copy(), top, left)
                prior_imgs.append(img)
                # else:
                #     if i < self.ext_frames:
                #         img = self._data_augmentation(self.time_encoding[i].copy(), top, left)
                #         prior_imgs.append(img)
        for i in range(self.dis_encoding.shape[0]):
            img = self._data_augmentation(self.dis_encoding[i].copy(), top, left)
            all_time_rsc.append(img)
            if i % self.ext_frames == 0 or i % self.ext_frames == index or i % self.ext_frames == self.ext_frames-1:
                # img = self._data_augmentation(self.dis_encoding[i].copy(), top, left)
                time_rsc.append(img)
        rs_imgs = torch.stack(rs_imgs, dim=0)
        # gs_imgs = torch.stack(gs_imgs, dim=0)
        # fl_imgs = torch.stack(fl_imgs, dim=0)
        prior_imgs = torch.stack(prior_imgs, dim=0)  # t2b: index,8 ,  b2t: index, 8
        time_rsc = torch.stack(time_rsc, dim=0)  # t2b: 0, index,8 ,  b2t: 0, index, 8
        all_time_rsc = torch.stack(all_time_rsc, dim=0)
        return rs_imgs, prior_imgs, time_rsc, all_time_rsc, out_paths

    def _data_augmentation(self, img, top, left, flip=False):
        if img.ndim == 3:
            img = img[top: top + self.crop_h, left: left + self.crop_w]
            if flip:
                img = img[:, ::-1, :]
            img = np.ascontiguousarray(img.transpose((2, 0, 1)))
            img = torch.from_numpy(img).float()
        elif img.ndim == 4:
            img = img[:, top: top + self.crop_h, left: left + self.crop_w]
            if flip:
                img = img[:, :, ::-1, :]
            img = np.ascontiguousarray(img.transpose((0, 3, 1, 2)))
            img = torch.from_numpy(img).float()
        return img

    def visualize(self, idx):
        rs_imgs, gs_imgs, fl_imgs, prior_imgs = self[idx]
        rs_frames = rs_imgs.shape[0] // 2
        gs_frames = gs_imgs.shape[0]
        # show imgs for t2b mode
        imgs = []
        for i in range(rs_frames):
            img = rs_imgs[i].permute(1, 2, 0).numpy()
            imgs.append(img)
        imgs = np.concatenate(imgs, axis=1)
        cv2.imshow('rs t2b', imgs)
        # show imgs for b2t mode
        imgs, fimgs = [], []
        for i in range(rs_frames, int(2 * rs_frames)):
            img = rs_imgs[i].permute(1, 2, 0).numpy()
            imgs.append(img)
        imgs = np.concatenate(imgs, axis=1)
        cv2.imshow('rs b2t', imgs)
        # show gs imgs and flows
        imgs, t2b_flows, b2t_flows = [], [], []
        for i in range(gs_frames):
            img = gs_imgs[i].permute(1, 2, 0).numpy()
            t2b_flow = normalize(flow_to_image(fl_imgs[i].permute(1, 2, 0).numpy(), convert_to_bgr=True),
                                 normalize=self.normalize, centralize=self.centralize)
            b2t_flow = normalize(
                flow_to_image(fl_imgs[gs_imgs.shape[0] + i].permute(1, 2, 0).numpy(), convert_to_bgr=True),
                normalize=self.normalize, centralize=self.centralize)
            imgs.append(img)
            t2b_flows.append(t2b_flow)
            b2t_flows.append(b2t_flow)
        imgs = np.concatenate(imgs, axis=1)
        t2b_flows = np.concatenate(t2b_flows, axis=1)
        b2t_flows = np.concatenate(b2t_flows, axis=1)
        cv2.imshow('gs, t2b flows, b2t flows', np.concatenate((imgs, t2b_flows, b2t_flows), axis=0))
        # warped imgs according to t2b flows
        t2b_imgs = rs_imgs[rs_frames // 2].repeat(gs_frames, 1, 1, 1)
        t2b_flows = fl_imgs[:gs_frames]
        warped_imgs = warp(t2b_imgs, t2b_flows, device='cpu')
        n, c, h, w = warped_imgs.shape
        warped_imgs = warped_imgs.permute(2, 0, 3, 1).reshape(h, n * w, c).numpy()
        cv2.imshow('warped t2b rs', warped_imgs)
        # warped imgs according to b2t flows
        b2t_imgs = rs_imgs[rs_frames + rs_frames // 2].repeat(gs_frames, 1, 1, 1)
        b2t_flows = fl_imgs[gs_frames:]
        warped_imgs = warp(b2t_imgs, b2t_flows, device='cpu')
        n, c, h, w = warped_imgs.shape
        warped_imgs = warped_imgs.permute(2, 0, 3, 1).reshape(h, n * w, c).numpy()
        cv2.imshow('warped b2t rs', warped_imgs)
        # show distortion t2b prior maps
        imgs = []
        for img in prior_imgs[:gs_frames]:
            img = (img[0].numpy() + 1) * (255. / 2)
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            imgs.append(img)
        imgs = np.concatenate(imgs, axis=1)
        cv2.imshow('t2b prior maps', imgs)
        # show distortion b2t prior maps
        imgs = []
        for img in prior_imgs[gs_frames:]:
            img = (img[0].numpy() + 1) * (255. / 2)
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            imgs.append(img)
        imgs = np.concatenate(imgs, axis=1)
        cv2.imshow('b2t prior maps', imgs)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# class Dataloader:
#     def __init__(self, para, ds_type='train'):
#         path = join(para.data_root, para.dataset, ds_type)
#         flow = True
#         if not 'EPE' in para.loss:
#             flow = False
#         dataset = RSGOPRO(path, para.future_frames, para.past_frames, para.frames, para.patch_size, para.centralize,
#                           para.normalize, flow)
#         bs = para.batch_size
#         ds_len = len(dataset)
#         self.loader = DataLoader(
#             dataset=dataset,
#             batch_size=para.batch_size,
#             shuffle=True,
#             num_workers=para.threads,
#             pin_memory=True,
#             drop_last=True
#         )
#         self.loader_len = int(np.ceil(ds_len / bs) * bs)
#
#     def __iter__(self):
#         return iter(self.loader)
#
#     def __len__(self):
#         return self.loader_len


# if __name__ == '__main__':
#     # path = '/home/zhong/Dataset/RS-GOPRO_DS/test/'
#     # ds = RSGOPRO(path=path, crop_size=(256, 256), ext_frames=5)
#     # idx = random.randint(0, len(ds) - 1)
#     # ds.visualize(idx)
#     import sys
#
#     sys.path.append('..')
#     from para import Parameter
#
#     args = Parameter().args
#     args.dataset = 'RS-GOPRO_DS'
#     args.frames = 5
#     args.patch_size = [256, 256]
#     dl = Dataloader(args)
#     count = 10
#     start_time = time()
#     for x, y, z, p in dl:
#         print(x.shape, y.shape, z.shape, p.shape)
#         count -= 1
#         if count == 0:
#             break
#     print(time() - start_time)
