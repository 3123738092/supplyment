import importlib
import json
import os
import cv2
from scipy.ndimage.filters import convolve, convolve1d
import numpy as np


def _cnv_h(x, y):
    """Perform horizontal convolution."""

    return convolve1d(x, y, mode="mirror")


def _cnv_v(x, y):
    """Perform vertical convolution."""

    return convolve1d(x, y, mode="mirror", axis=0)


def get_ch_masks_of_CFA(bayer, cfa_pattern):
    R_m = np.zeros_like(bayer)
    G_m = np.zeros_like(bayer)
    B_m = np.zeros_like(bayer)
    for i in range(2):
        for j in range(2):
            if cfa_pattern[i * 2 + j] == 'R':
                R_m[i::2, j::2] = 1
            elif cfa_pattern[i * 2 + j] == 'G':
                G_m[i::2, j::2] = 1
            else:
                B_m[i::2, j::2] = 1
    return R_m, G_m, B_m


def demosaicing_CFA_Bayer(CFA, pattern):
    R_m, G_m, B_m = get_ch_masks_of_CFA(CFA, pattern)

    h_0 = np.array([-0.25, 0.5, 0.5, 0.5, -0.25])

    R = CFA * R_m
    G = CFA * G_m
    B = CFA * B_m

    G_H = np.where(G_m == 0, _cnv_h(CFA, h_0), G)
    G_V = np.where(G_m == 0, _cnv_v(CFA, h_0), G)

    C_H = np.where(R_m == 1, R - G_H, 0)
    C_H = np.where(B_m == 1, B - G_H, C_H)

    C_V = np.where(R_m == 1, R - G_V, 0)
    C_V = np.where(B_m == 1, B - G_V, C_V)

    D_H = np.abs(C_H - np.pad(C_H, ((0, 0), (0, 2)), mode="reflect")[:, 2:])
    D_V = np.abs(C_V - np.pad(C_V, ((0, 2), (0, 0)), mode="reflect")[2:, :])

    del h_0, CFA, C_V, C_H

    k = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
        ]
    )

    d_H = convolve(D_H, k, mode="constant")
    d_V = convolve(D_V, np.transpose(k), mode="constant")

    del D_H, D_V

    mask = d_V >= d_H
    G = np.where(mask, G_H, G_V)
    M = np.where(mask, 1, 0)

    # k2 = np.array([0.1, 0.8, 0.1])
    # G = _cnv_h(G, k2)
    # G = _cnv_v(G, k2)

    k2 = np.array([[0.05, 0.05, 0.05],
                   [0.05, 0.6, 0.05],
                   [0.05, 0.05, 0.05]])
    G = convolve(G, k2, mode="reflect")

    del d_H, d_V, G_H, G_V

    # Red rows.
    R_r = np.transpose(np.any(R_m == 1, axis=1)[None]) * np.ones(R.shape)
    # Blue rows.
    B_r = np.transpose(np.any(B_m == 1, axis=1)[None]) * np.ones(B.shape)

    k_b = np.array([0.5, 0.0, 0.5])

    # Gr --> R
    R = np.where(
        np.logical_and(G_m == 1, R_r == 1),
        G + _cnv_h(R, k_b) - _cnv_h(G, k_b),
        R,
    )
    # Gb --> R
    R = np.where(
        np.logical_and(G_m == 1, B_r == 1) == 1,
        G + _cnv_v(R, k_b) - _cnv_v(G, k_b),
        R,
    )
    # Gb --> B
    B = np.where(
        np.logical_and(G_m == 1, B_r == 1),
        G + _cnv_h(B, k_b) - _cnv_h(G, k_b),
        B,
    )
    # Gr --> B
    B = np.where(
        np.logical_and(G_m == 1, R_r == 1) == 1,
        G + _cnv_v(B, k_b) - _cnv_v(G, k_b),
        B,
    )
    # B --> R
    R = np.where(
        np.logical_and(B_r == 1, B_m == 1),
        np.where(
            M == 1,
            B + _cnv_h(R, k_b) - _cnv_h(B, k_b),
            B + _cnv_v(R, k_b) - _cnv_v(B, k_b),
        ),
        R,
    )
    # R --> B
    B = np.where(
        np.logical_and(R_r == 1, R_m == 1),
        np.where(
            M == 1,
            R + _cnv_h(B, k_b) - _cnv_h(R, k_b),
            R + _cnv_v(B, k_b) - _cnv_v(R, k_b),
        ),
        B,
    )

    RGB = np.dstack((R, G, B))

    del R, G, B, k_b, R_r, B_r

    return RGB


def apply_awb(raw_or_rgb, awb_gain):
    awb_gain = np.expand_dims(awb_gain, axis=1)
    out = np.zeros_like(raw_or_rgb)
    out[:, :, ::2, ::2] = raw_or_rgb[:, :, ::2, ::2] * awb_gain[:, :, :, 0]
    out[:, :, 1::2, 1::2] = raw_or_rgb[:, :, 1::2, 1::2] * awb_gain[:, :, :, 2]
    out[:, :, ::2, 1::2] = raw_or_rgb[:, :, ::2, 1::2] * awb_gain[:, :, :, 1]
    out[:, :, 1::2, ::2] = raw_or_rgb[:, :, 1::2, ::2] * awb_gain[:, :, :, 1]
    out = np.clip(out, 0., 1.)
    return out


def autogain_bayer(raw_in):
    R = np.mean(raw_in[::2, ::2])
    G = (np.mean(raw_in[1::2, ::2])) / 2. + (np.mean(raw_in[::2, 1::2])) / 2.
    B = np.mean(raw_in[1::2, 1::2])
    return G / R, G / B


def demosaic_Menon2007(bayer, cfa='RGGB'):
    bayer_np = bayer[0, 0]
    rgb_np = demosaicing_CFA_Bayer(bayer_np, pattern=cfa)
    rgb_np = np.expand_dims(np.transpose(rgb_np, (2, 0, 1)), axis=0)
    return rgb_np


def apply_ccm(image, cam2rgbs):
    # Permute dimensions from (N, C, H, W) to (N, H, W, C)
    image = np.transpose(image, (0, 2, 3, 1))
    shape = image.shape
    image = image.reshape(-1, 3, 1)

    # Expand cam2rgbs to match the image size
    cam2rgbs = np.repeat(cam2rgbs, shape[1] * shape[2], axis=1)
    cam2rgbs = cam2rgbs.reshape(-1, 3, 3)

    # Perform batch matrix multiplication
    image = np.matmul(cam2rgbs, image)
    image = image.squeeze(-1).reshape(shape)

    # Permute dimensions back to (N, C, H, W)
    image = np.transpose(image, (0, 3, 1, 2))

    # Clamp values to the range [0, 1]
    image = np.clip(image, 0., 1.)

    return image


def process(raw, awb_gain, cam2rgb, pattern='rggb', gamma=2.2):
    raw = np.clip(raw[np.newaxis, np.newaxis, :], 0., 1.)
    if np.array_equal(awb_gain, np.ones((1, 1, 3))):
        awb_gain[0, 0, 0], awb_gain[0, 0, 2] = autogain_bayer(raw[0, 0])
    raw = apply_awb(raw, awb_gain)

    rgb = demosaic_Menon2007(np.clip(raw, 1e-4, 1))
    rgb = apply_ccm(rgb, cam2rgb)
    rgb = np.power(rgb, 1/gamma)

    rgb = np.clip(rgb, 0., 1.)
    rgb = np.transpose(rgb[0], (1, 2, 0))
    return rgb


if __name__ == '__main__':
    print(1)
