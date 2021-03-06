# -*- coding: utf-8 -*-
import os
import pathlib
import json
import numpy as np
import pandas as pd
from skimage import io
from random import randint
from scipy.ndimage.measurements import label

from imutils import normalize_by_lung_quantiles, normalize_by_lung_mean_std, normalize_by_lung_convex_hull
from imutils import imresize, augment_image_hard, normalize_by_histeq
from batch_reader import BatchReader


def process_ith_augmented(class_number, filename, i, imgs, is_val, masks, out_img_dir, train_classes, train_paths,
                          val_classes, val_paths, to_noise):
    img = imgs[i]
    mask = masks[i]

    if to_noise:
        lung_intensities = img.flatten()
        lung_intensities = lung_intensities[mask.flatten() > 0.5]
        mean_intensity = np.mean(lung_intensities)
        std_intensity = np.std(lung_intensities)

        noise = np.random.normal(mean_intensity, std_intensity, img.shape)
        segmented = img * mask + noise * (1 - mask)
    else:
        segmented = img

    segmented[segmented < 0] = 0
    segmented[segmented > 1] = 1

    out_img_filename = '%06i_%s_aug%03i.png' % (randint(0, 1e6 - 1), filename, i)
    out_img_path = os.path.join(out_img_dir, out_img_filename)
    io.imsave(out_img_path, (segmented * 255).astype(np.uint8))

    if not is_val:
        train_paths.append(out_img_path)
        train_classes.append(class_number)
    else:
        val_paths.append(out_img_path)
        val_classes.append(class_number)


def process_and_validate_mask(mask):
    bw = mask > 0.5

    labels, num_connected_components = label(bw)

    if num_connected_components < 2:
        return None

    size_idx = []
    for l in range(1, num_connected_components + 1):
        size = np.sum(labels == l)
        size_idx.append([size, l])

    size_idx.sort(reverse=True)

    lung_areas_ratio = size_idx[1][0] / size_idx[0][0]
    min_lung_areas_ratio = 0.55
    if lung_areas_ratio < min_lung_areas_ratio:
        return None

    bw = bw * 0
    for i in range(2):
        l = size_idx[i][1]
        bw += labels == l

    return bw


def get_cropping(mask_bw):
    proj_x = np.sum(mask_bw, axis=0).flatten()
    proj_y = np.sum(mask_bw, axis=1).flatten()

    d = min(mask_bw.shape) // 20
    x_low = max(0, np.where(proj_x > 0)[0][0] - d)
    x_high = min(mask_bw.shape[1], np.where(proj_x > 0)[0][-1] + d)
    y_low = max(0, np.where(proj_y > 0)[0][0] - d)
    y_high = min(mask_bw.shape[0], np.where(proj_y > 0)[0][-1] + d)

    return x_low, x_high, y_low, y_high


def find_file(img_path, data_dirs):
    for data_dir in data_dirs:
        full_path = os.path.join(data_dir, img_path)

        if os.path.isfile(full_path):
            return full_path

    return None


def process_row(out_img_dir, row, data_dirs, to_augment, train_classes, train_paths, val_classes, val_paths,
                batch_reader, to_crop, to_noise, out_size, normalization):
    img_path = row[1]['path']
    filename = row[1]['filename']
    class_number = row[1]['class_number']
    is_val = row[1]['is_val']

    img_path = find_file(img_path, data_dirs)
    mask0 = batch_reader.get_mask_of(filename)

    if os.path.isfile(img_path) and mask0 is not None:
        img0 = io.imread(img_path).astype(float)

        mask = process_and_validate_mask(mask0)

        if mask is not None:
            mask = imresize(mask, img0.shape, order=0)

            if to_crop:
                x_low, x_high, y_low, y_high = get_cropping(mask)
                img = img0[y_low:y_high, x_low:x_high]
                mask = mask[y_low:y_high, x_low:x_high]
            else:
                img = img0

            img = normalization(img, mask)

            out_shape = (out_size, out_size)
            img = imresize(img, out_shape)
            mask = imresize(mask, out_shape, order=0)

            if to_augment and (not is_val):
                imgs, masks = augment_image_hard(img, mask, num_copies=20)
            else:
                imgs = [img]
                masks = [mask]

            for i in range(len(imgs)):
                process_ith_augmented(class_number, filename, i, imgs, is_val, masks, out_img_dir, train_classes,
                                      train_paths, val_classes, val_paths, to_noise)


def prepare_dataset():
    data_dirs = ['e:/', 'f:/']

    class_of_interest = 'tuberculosis'
    # class_of_interest = 'abnormal_lungs'
    ver = 'v2.5'

    to_augment = False
    to_crop = True
    to_noise = True
    out_size = 512
    normalization = normalize_by_lung_convex_hull

    out_dir = os.path.join('d:/DATA/PTD/new/', class_of_interest, ver)

    out_img_dir = os.path.join(out_dir, 'img')
    print('Making dir ' + out_img_dir)
    pathlib.Path(out_img_dir).mkdir(parents=True, exist_ok=True)

    batch_reader = BatchReader(os.path.join(out_dir, 'img_previews'))

    study_group_filepath = '../data/study_group_class_' + class_of_interest + '.txt'
    df = pd.read_csv(study_group_filepath)

    train_paths = []
    train_classes = []
    val_paths = []
    val_classes = []
    print('Iterating rows of ' + study_group_filepath)
    for row in df.iterrows():
        i = row[0]
        if i % 100 == 0:
            print('%i / %i' % (i, df.shape[0]))

        process_row(out_img_dir, row, data_dirs, to_augment, train_classes, train_paths, val_classes, val_paths,
                    batch_reader, to_crop, to_noise, out_size, normalization)

    config_file_path = os.path.join(out_dir, 'config.json')
    data = {'to_augment': to_augment, 'to_crop': to_crop, 'to_noise': to_noise, 'out_size': out_size,
            'normalization': normalization.__name__, 'dataset': class_of_interest + '_' + ver,
            'train_cases': len(train_classes), 'val_cases': len(val_classes)}
    with open(config_file_path, 'wt') as f:
        json.dump(data, f)

    fn = os.path.join(out_dir, 'train.txt')
    print('Writing training data to ' + fn)
    df = pd.DataFrame.from_dict({'a_paths': train_paths, 'b_classes': train_classes})
    df.to_csv(fn, index=False, header=False, sep=' ')

    fn = os.path.join(out_dir, 'val.txt')
    print('Writing validation data to ' + fn)
    df = pd.DataFrame.from_dict({'a_paths': val_paths, 'b_classes': val_classes})
    df.to_csv(fn, index=False, header=False, sep=' ')

    fn = os.path.join(out_dir, 'class_labels.txt')
    print('Writing data to ' + fn)
    df = pd.DataFrame.from_dict({'a': ['norm', class_of_interest]})
    df.to_csv(fn, index=False, header=False)


if __name__ == '__main__':
    prepare_dataset()
