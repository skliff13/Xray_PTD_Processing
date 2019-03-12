import os
import json
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from xray_predictor import XrayPredictor
from imutils import to_uint8


def main():
    warnings.filterwarnings('ignore')

    xp = XrayPredictor('setup_vgg16_1.json')

    input_image_path = 'test_data/val/tuberculosis_03.png'

    predictions, rgb, img_normalized = xp.load_and_predict_image(input_image_path)

    img = xp.img_original.astype(float)
    img = (img - img.min()) / (img.max() - img.min())

    io.imsave('1_original.png', to_uint8(img))
    io.imsave('2_lungs.png', to_uint8(xp.mask))
    io.imsave('3_normalized.png', to_uint8(img_normalized))
    io.imsave('4_roi.png', to_uint8(xp.img_roi))

    # io.imshow_collection((img, xp.mask, img_normalized, xp.img_roi), cmap='gray')
    # io.show()


if __name__ == '__main__':
    main()
