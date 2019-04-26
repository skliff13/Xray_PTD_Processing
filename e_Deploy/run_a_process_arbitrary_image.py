import os
import json
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from glob import glob

# from xray_predictor import XrayPredictor
from xray_predictor_multi import XrayPredictorMulti


def pred2str(predictions, items_per_row=3):
    rows = []

    i = 0
    row = ''
    for class_name in predictions:
        row += class_name + '=' + str(predictions[class_name])
        i += 1

        if i % items_per_row !=0:
            row += ', '
        else:
            rows.append(row)
            row = ''

    if row:
        rows.append(row)

    return '\n'.join(rows)


def save_or_plot_combined_image(elapsed, img_normalized, input_image_path, predictions, rgb, to_plot):
    tmp = rgb * 0
    for c in range(3):
        tmp[:, :, c] = img_normalized
    combined = np.concatenate((tmp, rgb), axis=1)
    prob = predictions['abnormal_lungs']
    if not to_plot:
        out_path = '%s+heat-abnorm%.02f.png' % (input_image_path, prob)
        io.imsave(out_path, combined)

        out_path = '%s+heat-abnorm%.02f.json' % (input_image_path, prob)
        with open(out_path, 'w') as f:
            json.dump(predictions, f, indent=2)
    else:
        plt.imshow(combined)
        title = '%s+heat (%.2f sec)\n' % (input_image_path, elapsed)
        title += pred2str(predictions)
        plt.title(title)
        out_path = '%s+plot-abnorm%.02f.png' % (input_image_path, prob)
        plt.savefig(out_path)
        plt.pause(0.1)


def process_image_dir(dir_with_images, to_plot, xp):
    files = os.listdir(dir_with_images)
    for file in files:
        input_image_path = os.path.join(dir_with_images, file)
        if '+' not in file and os.path.isfile(input_image_path) and input_image_path.endswith('.png'):
            start = time.time()
            predictions, rgb, img_normalized = xp.load_and_predict_image(input_image_path)
            elapsed = time.time() - start
            print('Time elapsed: %.02f sec' % elapsed)

            save_or_plot_combined_image(elapsed, img_normalized, input_image_path, predictions, rgb, to_plot)


def main():
    warnings.filterwarnings('ignore')

    # xp = XrayPredictor('setup_vgg19_1.json')
    xp = XrayPredictorMulti('setup_vgg16m_1.json')
    to_plot = False
    plt.figure(figsize=(10, 7))

    meta_dir = '/hdd_purple/ImageGeneration/Xray/generated_per_age_by_10k/unpacked'
    dirs = glob(os.path.join(meta_dir, 'gen_*'))
    dirs.sort()

    for dir_with_images in dirs:
        process_image_dir(dir_with_images, to_plot, xp)


if __name__ == '__main__':
    main()
