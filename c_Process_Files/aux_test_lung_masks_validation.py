# -*- coding: utf-8 -*-
import os
from glob import glob
from skimage import io
from run_c_prepare_dataset import process_and_validate_mask

def check_lung_masks_validation():
    dir_with_segmented_xrays = '../data/xray_test/'

    dump_dir = os.path.join(dir_with_segmented_xrays, '_dumped')
    if not os.path.isdir(dump_dir):
        os.mkdir(dump_dir)

    mask_files = glob(dir_with_segmented_xrays + '/*-mask.png')
    for mask_file in mask_files:
        mask = io.imread(mask_file)

        if process_and_validate_mask(mask) is None:
            out_path = os.path.join(dump_dir, os.path.split(mask_file)[1])
            print('Writing dumped to "%s"' % out_path)
            io.imsave(out_path, mask)


if __name__ == '__main__':
    check_lung_masks_validation()
