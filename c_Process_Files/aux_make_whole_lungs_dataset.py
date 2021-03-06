# -*- coding: utf-8 -*-
import os
import pathlib
import numpy as np
import pandas as pd
from glob import glob
from skimage import io
from random import randint

from imutils import imresize, augment_image


def process_ith_augmented(class_number, filename, i, imgs, is_val, masks, out_img_dir, train_classes, train_paths,
                          val_classes, val_paths):
    img = imgs[i]
    mask = masks[i]

    segm_bw = mask > 0.5
    lung_intensities = img.flatten()[segm_bw.flatten() > 0.5]
    mean_intensity = np.mean(lung_intensities)
    std_intensity = np.std(lung_intensities)

    noise = np.random.normal(mean_intensity, std_intensity, img.shape)
    segmented = img * mask + noise * (1 - mask)
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


def process_row(out_img_dir, row, src_dirs, to_augment, train_classes, train_paths, val_classes, val_paths):
    img_path = row[1]['path']
    filename = row[1]['filename']
    class_number = row[1]['class_number']
    is_val = row[1]['is_val']

    mask_path = os.path.join(src_dirs[class_number], os.path.split(img_path)[0], 'comp02*' + filename)
    found_files = glob(mask_path)

    if found_files:
        rgb = io.imread(found_files[0]).astype(float) / 255.
        img = rgb[:, :, 1]
        mask = rgb[:, :, 0]

        if to_augment and (not is_val):
            imgs, masks = augment_image(img, mask)
        else:
            imgs = [img]
            masks = [mask]

        for i in range(len(imgs)):
            process_ith_augmented(class_number, filename, i, imgs, is_val, masks, out_img_dir, train_classes,
                                  train_paths, val_classes, val_paths)


def make_whole_lungs_dataset():
    norm_dir = 'd:/IMG/XRay_Obvodki_b_Misuk_FULL_incl_PNGs_of_Lungs/00_Scripts/PTD1_Norm/'
    class_dir = 'd:/IMG/XRay_Obvodki_b_Misuk_FULL_incl_PNGs_of_Lungs/00_Scripts/klassi/combined_abnormal_lungs/'

    class_of_interest = 'tuberculosis'
    # class_of_interest = 'abnormal_lungs'
    to_augment = True

    out_dir = os.path.join('d:/DATA/PTD/new/', class_of_interest, 'v1.2')

    out_img_dir = os.path.join(out_dir, 'img')
    print('Making dir ' + out_img_dir)
    pathlib.Path(out_img_dir).mkdir(parents=True, exist_ok=True)

    study_group_filepath = '../data/study_group_class_' + class_of_interest + '.txt'
    df = pd.read_csv(study_group_filepath)

    src_dirs = [norm_dir, class_dir]

    train_paths = []
    train_classes = []
    val_paths = []
    val_classes = []
    for row in df.iterrows():
        i = row[0]
        if i % 100 == 0:
            print('%i / %i' % (i, df.shape[0]))

        process_row(out_img_dir, row, src_dirs, to_augment, train_classes, train_paths, val_classes, val_paths)

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
    make_whole_lungs_dataset()
