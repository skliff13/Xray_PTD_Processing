# -*- coding: utf-8 -*-
import numpy as np
from random import randint
from skimage import transform, exposure
from skimage.morphology import convex_hull_image


def imresize(m, new_shape, order=1, mode='constant'):
    dtype = m.dtype
    mult = np.max(np.abs(m)) * 2
    m = m.astype(np.float32) / mult
    m = transform.resize(m, new_shape, order=order, mode=mode)
    m = m * mult
    return m.astype(dtype=dtype)


def normalize_by_lung_quantiles(img, mask):
    lung_intensities = img.flatten()
    lung_intensities = lung_intensities[mask.flatten() > 0]
    q1 = np.percentile(lung_intensities, 1)
    q2 = np.percentile(lung_intensities, 99)
    img = 0.2 + 0.6 * (img - q1) / (q2 - q1)
    return img


def normalize_by_lung_mean_std(img, mask):
    lung_intensities = img.flatten()
    lung_intensities = lung_intensities[mask.flatten() > 0]
    mn = lung_intensities.mean()
    std = lung_intensities.std()
    img -= mn
    img /= std * 7.
    img += 0.5
    return img


def normalize_by_lung_convex_hull(img, mask):
    hull = convex_hull_image(imresize(mask, (256, 256)))
    hull = imresize(hull, mask.shape)
    lung_intensities = img.flatten()
    lung_intensities = lung_intensities[hull.flatten() > 0]
    q1 = np.percentile(lung_intensities, 1)
    q2 = np.percentile(lung_intensities, 99)
    img = 0.2 + 0.6 * (img - q1) / (q2 - q1)
    return img


def normalize_by_histeq(img):
    return exposure.equalize_hist(img)


def to_uint8(img):
    img[img < 0] = 0
    img[img > 1] = 1
    return (img * 255).astype(np.uint8)

