import torch
import math
import numpy as np

def generate_2D_grid(H, W):
    x = torch.arange(0, W, 1).float()
    y = torch.arange(0, H, 1).float()

    xx = x.repeat(H, 1)
    yy = y.view(H, 1).repeat(1, W)

    grid = torch.stack([xx, yy], dim=0)

    return grid  # (2,H,W)


def distortion_map(h, w, ref_row, reverse=False):
    grid_row = generate_2D_grid(h, w)[1].float()
    mask = grid_row / (h - 1)
    if reverse:
        mask *= -1.
        ref_row_floor = math.floor(h - 1 - ref_row)
        mask = mask - mask[int(ref_row_floor)] + (h - 1 - ref_row - ref_row_floor) * (1. / (h - 1))
    else:
        ref_row_floor = math.floor(ref_row)
        mask = mask - mask[int(ref_row_floor)] - (ref_row - ref_row_floor) * (1. / (h - 1))

    return mask

def self_distortion_map(h, w, reverse=False, is_zero=False):
    if is_zero:
        return torch.zeros(h, w).float()
    y = torch.arange(-(h-1), h, 2).float()
    yy = y.view(h, 1).repeat(1, w)
    mask = yy / (h - 1)
    if reverse:
        mask *= -1.
    return mask

def gs_time_code(h, w, ref_row1, ref_row2):
    ref_row_floor1, ref_row_floor2 = math.floor(ref_row1), math.floor(ref_row2)
    mask = torch.zeros(h, w).float()
    mask[int(ref_row_floor1):int(ref_row_floor2)] = 1.   #+1
    grid_row = generate_2D_grid(h, w)[1].float()
    gs_time_map = grid_row / (h - 1)
    min = gs_time_map[int(ref_row_floor1)][0] + (ref_row1 - ref_row_floor1) * (1. / (h - 1))
    max = gs_time_map[int(ref_row_floor2)][0] + (ref_row2 - ref_row_floor2) * (1. / (h - 1))
    gs_time_map = (gs_time_map - min) / (max - min)
    gs_time_map = gs_time_map * mask

    return gs_time_map[..., np.newaxis], mask[..., np.newaxis]

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

    return np.stack([formask[..., np.newaxis], backmask[..., np.newaxis], warpmask[..., np.newaxis]], axis=0)

def time_map_rsc(h, w, ref_row, reverse=False):
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
        formask = 1 - formask

        backmask[backmask < 1] = -1
        backmask[backmask >= 1] -= 1
        backmask = backmask / backmask.max()
        backmask[backmask < 0] = 0
        backmask = torch.flip(backmask, [0])
        warpmask = torch.flip(warpmask, [0])

    else:
        formask[formask > 1] = 1
        formask = 1 - formask
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

    return np.stack([formask[..., np.newaxis], backmask[..., np.newaxis], warpmask[..., np.newaxis]], axis=0)

def distortion_encoding(x, ref_row):
    assert x.dim() == 4
    n, c, h, w = x.shape
    grid_row = generate_2D_grid(h, w)[1].float()
    mask = grid_row / (h - 1)
    ref_row_floor = math.floor(ref_row)
    mask = mask - mask[int(ref_row_floor)] - (ref_row - ref_row_floor) * (1. / (h - 1))
    mask = mask.reshape(1, 1, h, w)
    x *= mask
    return x, mask


if __name__ == '__main__':
    # x = torch.ones(1, 1, 256, 256).cuda() * 2
    # out, mask = distortion_encoding(x, ref_row=0)
    # print('out:', out)
    # print('mask', mask)
    # x = torch.ones(1, 1, 256, 256).cuda() * 2
    # out, mask = distortion_encoding(x, ref_row=255 / 2)
    # print('out:', out)
    # print('mask', mask)
    # x = torch.ones(1, 1, 256, 256).cuda() * 2
    # out, mask = distortion_encoding(x, ref_row=255)
    # print('out:', out)
    # print('mask', mask)
    ## forward: scanning from up to bottom
    mask = distortion_map(h=256, w=256, ref_row=0)
    print('mask', mask)
    mask = distortion_map(h=256, w=256, ref_row=255 / 2)
    print('mask', mask)
    mask = distortion_map(h=256, w=256, ref_row=255)
    print('mask', mask)
    ## reverse: scanning from bottom to up
    mask = distortion_map(h=256, w=256, ref_row=0, reverse=True)
    print('mask', mask)
    mask = distortion_map(h=256, w=256, ref_row=255 / 2, reverse=True)
    print('mask', mask)
    mask = distortion_map(h=256, w=256, ref_row=255, reverse=True)
    print('mask', mask)
