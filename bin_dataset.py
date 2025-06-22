import random    # demosaic_hybridevs_dataset.py所需库
from glob import glob
from os.path import exists, join

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from absl.logging import info, warning

from torch.utils import data as data
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import os
from pathlib import Path
import random
import numpy as np
import torch

from enum import Enum    # DemosaicHybridevsBatch所需

def get_demosaic_hybridevs_dataset(opt): # the opt here is the opt['datasets'] in corresponding yml file，但是yml需要新建，且写yml时需要对照参数写入
    root = opt.root
    assert opt.training_set_rate >= 0.1 and opt.training_set_rate <= 0.99

    raw_root = join(root, "input")
    gt_root = join(root, "gt")
    raws = sorted(glob(join(raw_root, "*.bin")))
    gts = sorted(glob(join(gt_root, "*.png")))
    assert len(raws) == len(gts)

    training_length = int(len(raws) * opt.training_set_rate)
    train_raws = raws[:training_length]
    train_gts = gts[:training_length]
    test_raws = raws[training_length:]
    test_gts = gts[training_length:]
    train = DemosaicHybridevsDataset(
        is_train=True,
        is_random_crop=opt.is_random_crop,
        is_flip=opt.is_flip,
        is_rotation=opt.is_rotation,
        is_adding_noise=opt.is_adding_noise,
        noise_threshold=opt.noise_threshold,
        is_adding_defect_pixel=opt.is_adding_defect_pixel,
        positive_defect_rate=opt.positive_defect_rate,
        negative_defect_rate=opt.negative_defect_rate,
        raws=train_raws,
        gts=train_gts,
        random_crop_resolution=opt.random_crop_resolution,
        is_raw_from_vis=opt.is_raw_from_vis,
    )
    test = DemosaicHybridevsDataset(
        is_train=False,
        is_random_crop=True,
        is_flip=False,
        is_rotation=False,
        is_adding_noise=False,
        noise_threshold=0,
        is_adding_defect_pixel=False,
        positive_defect_rate=1,
        negative_defect_rate=0,
        raws=test_raws,
        gts=test_gts,
        random_crop_resolution=opt.random_crop_resolution,
        is_raw_from_vis=opt.is_raw_from_vis,
    )
    return train, test


def get_demosaic_hybridevs_val_dataset(opt):
    root = opt.root
    is_raw_from_vis = opt.is_raw_from_vis
    is_random_crop = opt.is_random_crop
    random_crop_resolution = opt.random_crop_resolution

    raw_root = join(root, "input")
    raws = sorted(glob(join(raw_root, "*.bin")))
    fake_gt = sorted(glob(join(raw_root, "*-raw.png")))
    test = DemosaicHybridevsDataset(
        is_train=False,
        is_random_crop=is_random_crop,
        is_flip=False,
        is_rotation=False,
        is_adding_noise=False,
        noise_threshold=0,
        is_adding_defect_pixel=False,
        positive_defect_rate=1,
        negative_defect_rate=0,
        raws=raws,
        gts=fake_gt,
        random_crop_resolution=random_crop_resolution,
        is_raw_from_vis=is_raw_from_vis,
    )
    return test, test

def read_aps(dp):
    aps = np.fromfile(dp, np.uint16).reshape(720, 1280)
    return aps

def decompress_from_2bit(compressed):
    unpacked = np.zeros(compressed.size * 4, dtype=np.uint8)
    unpacked[::4] = (compressed >> 6) & 0b11
    unpacked[1::4] = (compressed >> 4) & 0b11
    unpacked[2::4] = (compressed >> 2) & 0b11
    unpacked[3::4] = compressed & 0b11

    reverse_mapping = {0: -1, 1: 0, 2: 1}
    decoded = np.vectorize(reverse_mapping.get)(unpacked)

    return decoded.reshape((1, -1, 360, 640))


def read_evs(dp):
    evs_compress = np.fromfile(dp, dtype=np.int8)
    evs = decompress_from_2bit(evs_compress)
    return evs


def binning_frame(evs_frame, out_frame=8):
    b, c, h, w = evs_frame.shape
    out_evs = torch.zeros((b, out_frame, h, w))
    for i in range(out_frame):
        start_num = c / out_frame * i
        end_num = c / out_frame * (i+1)

        k_start = int(np.floor(start_num))
        k_end = int(np.ceil(end_num))
        for k in range(k_start, k_end):
            if k < 0 or k >= c:
                continue
            seq_start = max(start_num, k)
            seq_end = min(end_num, k+1)
            weights = seq_end - seq_start
            out_evs[:, i, :, :] += evs_frame[:, k] * weights
    return out_evs


#从basic_batch.py中摘抄
class B(Enum):      # 原名DemosaicHybridevsBatch
    IMAGE_NAME = "image_name"
    HEIGHT = "height"
    WIDTH = "width"
    RAW_TENSOR = "raw_image"
    GROUND_TRUTH = "ground_truth"
    RAW_RGB_POSITION = "raw_rgb_position"
    PREDICTION = "prediction"

def get_demosaic_batch():
    batch = {}
    for item in B:
        batch[item] = "NONE(str)"
    return batch

