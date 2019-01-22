# -*- coding: utf-8 -*-
import os
import pathlib
import numpy as np
import pandas as pd
from glob import glob
from skimage import io, transform
from random import randint


def segment_lungs_caffe():
    class_of_interest = 'tuberculosis'
    # class_of_interest = 'abnormal_lungs'

    img_dir = os.path.join('d:/DATA/PTD/new/', class_of_interest, 'v1.3', 'img_previews')

    path_to_python_with_caffe = 'python.exe'
    path_to_job_model = 'lungs_segmentation_caffe/job_model_20190117-004144-1a46'
    path_to_script = 'lungs_segmentation_caffe/run02_caffe_segmentation_Inference_Simple_v2.py'

    os.system(' '.join([path_to_python_with_caffe, path_to_script, path_to_job_model, img_dir]))


if __name__ == '__main__':
    segment_lungs_caffe()
