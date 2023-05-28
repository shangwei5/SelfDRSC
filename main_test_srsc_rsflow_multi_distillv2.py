import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from models.select_model import define_Model

from data.dataset_rsgopro_self import RSGOPRO as D



def main(json_path='options/test_srsc_rsflow_multi_distillv2_psnr.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # # ----------------------------------------
    # # update opt
    # # ----------------------------------------
    # # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    # init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    # init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    # opt['path']['pretrained_netG'] = init_path_G
    # opt['path']['pretrained_netE'] = init_path_E
    # init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    # opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    # current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'test_110000'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # # ----------------------------------------
    # # seed
    # # ----------------------------------------
    # seed = opt['train']['manual_seed']
    # if seed is None:
    #     seed = random.randint(1, 10000)
    # print('Random seed: {}'.format(seed))
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # train_set = define_Dataset(dataset_opt)
            path = os.path.join(dataset_opt['data_root'], 'train')
            # flow = True
            # if not 'EPE' in para.loss:
            flow = False
            train_set = D(path, dataset_opt['future_frames'], dataset_opt['past_frames'], dataset_opt['frames'], dataset_opt['patch_size'], dataset_opt['centralize'],
                          dataset_opt['normalize'], flow, False)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            # test_set = define_Dataset(dataset_opt)
            path = os.path.join(dataset_opt['data_root'], 'test')  #'valid') #
            # flow = True
            # if not 'EPE' in para.loss:
            flow = False
            test_set = D(path, dataset_opt['future_frames'], dataset_opt['past_frames'], dataset_opt['frames'], None, dataset_opt['centralize'],
                          dataset_opt['normalize'], flow, False, True)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0

    for test_data in test_loader:
        idx += 1
        # print(test_data[5][0])
        video_name = test_data[7][0][0].split('/')[-3]

        img_dir = os.path.join(opt['path']['inference_results'], video_name)
        util.mkdir(img_dir)
        #rs_imgs, gs_imgs, fl_imgs, prior_imgs, is_b2t, out_paths, input_path
        # print(test_data[3].shape)
        model.feed_data(test_data)
        model.test()

        visuals = model.current_visuals()
        E_img = util.tensor2uint_list(visuals['E'])
        H_img = util.tensor2uint_list(visuals['H'])

        current_psnr = 0
        current_ssim = 0
        for save_idx in range(len(E_img)):
            image_name_ext = os.path.basename(test_data[7][save_idx][0])
            img_name, ext = os.path.splitext(image_name_ext)
            save_img_path = os.path.join(img_dir, '{:s}.png'.format(img_name))
            util.imsave(E_img[save_idx], save_img_path)
            current_psnr += util.calculate_psnr(E_img[save_idx], H_img[save_idx])
            current_ssim += util.calculate_ssim(E_img[save_idx], H_img[save_idx])
        # -----------------------
        # calculate PSNR
        # -----------------------
        # for j in range(len(E_img)):
        # print(E_img[j].shape, H_img[j].shape)
            # if j == 0:
        current_psnr = current_psnr / len(E_img)
        current_ssim = current_ssim / len(E_img)
            # else:
            #     current_psnr = current_psnr + util.calculate_psnr(E_img[j], H_img[j], border=border)
        # current_psnr = current_psnr / len(E_img)

        logger.info('{:->4d}--> {:>10s} | {:<4.3f}dB  {:.4f}'.format(idx, image_name_ext, current_psnr, current_ssim))

        avg_psnr += current_psnr
        avg_ssim += current_ssim

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx

    # testing log
    logger.info('Average PSNR : {:<.3f}dB\n'.format(avg_psnr))
    logger.info('Average SSIM : {:<.4f}\n'.format(avg_ssim))

if __name__ == '__main__':
    main()
