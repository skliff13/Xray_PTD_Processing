# -*- coding: utf-8 -*-
import os
import pathlib
import numpy as np
import pandas as pd
from glob import glob
from skimage import io, transform
from random import randint


def imresize(m, new_shape, order=1, mode='constant'):
    dtype = m.dtype
    mult = np.max(np.abs(m)) * 2
    m = m.astype(np.float32) / mult
    m = transform.resize(m, new_shape, order=order, mode=mode)
    m = m * mult
    return m.astype(dtype=dtype)


def process_row(out_img_dir, row, data_dir, preview_size):
    img_path = row[1]['path']
    filename = row[1]['filename']

    img_path = os.path.join(data_dir, img_path)
    found_files = glob(img_path)

    if found_files:
        img = io.imread(found_files[0]).astype(float)

        img = imresize(img, (preview_size, preview_size))

        box = [round(0.2539 * img.shape[0] - 1), round(0.4915 * img.shape[0] - 1)]
        intensities = img[box[0]:box[1], box[0]:box[1]].flatten()
        q1 = np.percentile(intensities, 5)
        q2 = np.percentile(intensities, 95)
        img = 0.2 + 0.6 * (img - q1) / (q2 - q1)

        img[img < 0] = 0
        img[img > 1] = 1
        img = (img * 255).astype(np.uint8)

        io.imsave(os.path.join(out_img_dir, filename), img)


def prepare_previews():
    data_dir = 'e:/'

    class_of_interest = 'tuberculosis'
    # class_of_interest = 'abnormal_lungs'
    preview_size = 256

    out_dir = os.path.join('d:/DATA/PTD/new/', class_of_interest, 'v1.3')

    out_img_dir = os.path.join(out_dir, 'img_previews')
    print('Making dir ' + out_img_dir)
    pathlib.Path(out_img_dir).mkdir(parents=True, exist_ok=True)

    study_group_filepath = '../data/study_group_class_' + class_of_interest + '.txt'
    df = pd.read_csv(study_group_filepath)

    for row in df.iterrows():
        i = row[0]
        if i % 100 == 0:
            print('%i / %i' % (i, df.shape[0]))

        process_row(out_img_dir, row, data_dir, preview_size)


if __name__ == '__main__':
    prepare_previews()
