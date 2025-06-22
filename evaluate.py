#!/usr/bin/env python
import numpy as np
import os.path
import sys
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import io, color
from datetime import datetime
import subprocess
from ISP import process


awb_gain = np.array([[[2.64267, 1., 1.763631]]])
cam2rgb = np.array([[[[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]]]])


truth_dir = os.path.join('/app/input/', 'ref')
submit_dir = os.path.join('/app/input/', 'res')
score_dir = '/app/output/'
shared_dir = '/app/shared/'

prediction_path = os.path.join(shared_dir, 'prediction')


def compare_mpsnr(x_true, x_pred, data_range):
    """
    :param x_true: Input image must have three dimension (H, W, C)
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    total_psnr = [compare_psnr(x_true[:, :, i], x_pred[:, :, i], data_range=data_range) for i in range(x_true.shape[2])]
    # total_psnr = [compare_psnr(x_true[::2, ::2], x_pred[::2, ::2], data_range=data_range),
    #               compare_psnr(x_true[1::2, ::2], x_pred[1::2, ::2], data_range=data_range),
    #               compare_psnr(x_true[::2, 1::2], x_pred[::2, 1::2], data_range=data_range),
    #               compare_psnr(x_true[1::2, 1::2], x_pred[1::2, 1::2], data_range=data_range)]

    return np.mean(total_psnr)


def compare_mssim(x_true, x_pred, data_range, multidimension=False):
    """

    :param x_true:
    :param x_pred:
    :param data_range:
    :param multidimension:
    :return:
    """
    mssim = [compare_ssim(x_true[:, :, i], x_pred[:, :, i], data_range=data_range) for i in range(x_true.shape[2])]
    # mssim = [compare_ssim(x_true[::2, ::2], x_pred[::2, ::2], data_range=data_range),
    #          compare_ssim(x_true[1::2, ::2], x_pred[1::2, ::2], data_range=data_range),
    #          compare_ssim(x_true[::2, 1::2], x_pred[::2, 1::2], data_range=data_range),
    #          compare_ssim(x_true[1::2, 1::2], x_pred[1::2, 1::2], data_range=data_range)]

    return np.mean(mssim)


output_filename = os.path.join(score_dir, 'scores.txt')

print(submit_dir)
print(truth_dir)
print(score_dir)

if not os.path.isdir(submit_dir):
    print("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)

# Get the path of gt file
img_list = []
# file_list = sorted(os.listdir(submit_dir))
file_list = sorted(os.listdir(truth_dir))

# set up the metris you need.
PSNRs = []
SSIMs = []

current_datetime = datetime.now()
current_date = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
print(f'Time: {current_date}\t Start evaluating')

for item in file_list:
    res_img = np.fromfile(os.path.join(submit_dir, item), np.uint16).reshape(720, 1280).astype(np.float32) / 1023
    ref_img = np.fromfile(os.path.join(truth_dir, item), np.uint16).reshape(720, 1280).astype(np.float32) / 1023
    res_rgb = process(res_img, awb_gain, cam2rgb, pattern='rggb', gamma=2.2)
    ref_rgb = process(ref_img, awb_gain, cam2rgb, pattern='rggb', gamma=2.2)
    PSNR = compare_mpsnr(res_rgb, ref_rgb, data_range=1)
    SSIM = compare_mssim(res_rgb, ref_rgb, data_range=1)
    PSNRs.append(PSNR)
    SSIMs.append(SSIM)
current_datetime = datetime.now()
current_date = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
print(f'Time: {current_date}\t End')
score_psnr = np.mean(PSNRs)
score_ssim = np.mean(SSIMs)

# Write the result into score_path/score.txt
with open(output_filename, 'w') as f:
    f.write('{}: {}\n'.format('PSNR', score_psnr))
    f.write('{}: {}\n'.format('SSIM', score_ssim))
    f.write('DEVICE: CPU\n')
