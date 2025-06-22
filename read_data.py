#保存在basicsr/data目录下

import os
import numpy as np
from omegaconf import OmegaConf
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
import torch


def demosaic_Menon2007(bayer, cfa='RGGB'):
    bayer_np = bayer[0, 0].cpu().numpy()
    rgb_np = demosaicing_CFA_Bayer_Menon2007(bayer_np, pattern=cfa)
    rgb_pt = torch.FloatTensor(rgb_np).permute(2, 0, 1).unsqueeze(0).to(bayer.device)
    return rgb_pt


def apply_awb(raw_or_rgb, awb_gain):
    awb_gain = awb_gain.unsqueeze(1)
    out = torch.zeros_like(raw_or_rgb)
    out[:, :, ::2, ::2] = raw_or_rgb[:, :, ::2, ::2] * awb_gain[:, :, :, 0]
    out[:, :, 1::2, 1::2] = raw_or_rgb[:, :, 1::2, 1::2] * awb_gain[:, :, :, 2]
    out[:, :, ::2, 1::2] = raw_or_rgb[:, :, ::2, 1::2] * awb_gain[:, :, :, 1]
    out[:, :, 1::2, ::2] = raw_or_rgb[:, :, 1::2, ::2] * awb_gain[:, :, :, 1]
    out = out.clamp(0., 1.)
    return out


def apply_ccm(image, cam2rgbs):
    image = image.permute(0, 2, 3, 1)
    shape = image.shape
    image = image.contiguous().view(-1, 3, 1)

    cam2rgbs = cam2rgbs.expand(-1, shape[1]*shape[2], -1, -1)
    cam2rgbs = cam2rgbs.contiguous().view(-1, 3, 3)

    image = torch.bmm(cam2rgbs, image)
    image = image.squeeze(-1).view(shape)
    image = image.permute(0, 3, 1, 2)

    image = image.clamp(0., 1.)

    return image


def process_isp(raw, conf):
    awb_gain = torch.FloatTensor(conf.simulator.meta.awb_gain).view(1, 1, 3)
    cam2rgb = torch.FloatTensor(conf.simulator.meta.cam2rgb).view(1, 1, 3, 3)
    raw = torch.FloatTensor(raw.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 1023
    raw = apply_awb(raw, awb_gain)

    rgb = demosaic_Menon2007(raw.clamp(1e-4,1))
    rgb = apply_ccm(rgb, cam2rgb)
    rgb1 = np.power(np.clip(rgb, 0, 1), 1/2.2)

    rgb1 = (rgb1*255.).clamp(0., 255.)
    return rgb1


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


if __name__ == '__main__':
    """
    aps_path: aps path (xxx/xxx.bin)
    evs_path: evs path (xxx/xxx.bin)
    conf_path: config path (isp conf)
    
    binning_frame: Convert the variable frame rate to a fixed frame rate.
    """
    aps_path = r'xxx/xxx.bin'
    evs_path = r'xxx/xxx.bin'
    conf_path = r'xxx/xxx.yaml'

    conf = OmegaConf.load(conf_path)
    # aps: uint 10, shape: (720, 1280)
    aps = read_aps(aps_path)
    # evs: int 8, shape: (-1, 360, 640)
    evs = read_evs(evs_path)

    rgb = process_isp(aps, conf)






