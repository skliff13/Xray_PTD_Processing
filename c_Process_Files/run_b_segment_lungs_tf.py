# -*- coding: utf-8 -*-
import os
import pathlib
import numpy as np
import pandas as pd
from glob import glob
from skimage import io, transform
from random import randint


def segment_lungs_tf():
    class_of_interest = 'pneumonia'
    # class_of_interest = 'tuberculosis'
    # class_of_interest = 'abnormal_lungs'

    img_dir = os.path.join('/home/skliff13/work/PTD_Xray/datasets', class_of_interest + '_list', 'img_previews')

    path_to_python2_with_tf = 'python'
    path_to_script = 'lungs_segmentation_tf/run_tf_segmentation.py'

    os.system(' '.join([path_to_python2_with_tf, path_to_script, img_dir]))


if __name__ == '__main__':
    segment_lungs_tf()
