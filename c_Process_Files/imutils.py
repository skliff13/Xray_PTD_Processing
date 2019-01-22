# -*- coding: utf-8 -*-
import numpy as np
from random import randint
from skimage import transform
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
    hull = convex_hull_image(mask)
    lung_intensities = img.flatten()
    lung_intensities = lung_intensities[hull.flatten() > 0]
    q1 = np.percentile(lung_intensities, 1)
    q2 = np.percentile(lung_intensities, 99)
    img = 0.2 + 0.6 * (img - q1) / (q2 - q1)
    return img


def to_uint8(img):
    img[img < 0] = 0
    img[img > 1] = 1
    return (img * 255).astype(np.uint8)


def augment_image(img, mask):
    imgs = [img]
    masks = [mask]

    r = 5
    dbs = [[0, 0, 0, 15], [0, 0, 10, 0], [10, 0, 0, 0], [10, 0, 0, 15], [0, 10, 0, 0],
           [0, 10, 10, 0], [10, 10, 0, 0], [10, 0, 10, 0], [0, 10, 0, 15], [0, 0, 10, 15]]

    for db in dbs:
        dx1 = db[0] + randint(0, r - 1)
        dy1 = db[1] + randint(0, r - 1)
        dx2 = db[2] + randint(0, r - 1)
        dy2 = db[3] + randint(0, r - 1)
        ii = [dy1, img.shape[0] - dy2]
        jj = [dx1, img.shape[0] - dx2]

        imgs.append(imresize(img[ii[0]:ii[1], jj[0]:jj[1]], img.shape))
        masks.append(imresize(mask[ii[0]:ii[1], jj[0]:jj[1]], img.shape))

        small_shape = (ii[1] - ii[0], jj[1] - jj[0])

        img1 = img * 0
        img1[ii[0]:ii[1], jj[0]:jj[1]] = imresize(img, small_shape)
        imgs.append(img1)

        mask1 = mask * 0
        mask1[ii[0]:ii[1], jj[0]:jj[1]] = imresize(mask, small_shape)
        masks.append(mask1)

    return imgs, masks
