import os
import json
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from xray_predictor import XrayPredictor


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


def main():
    warnings.filterwarnings('ignore')

    xp = XrayPredictor('setup_vgg16_1.json')

    to_plot = False
    plt.figure(figsize=(10, 7))

    files = os.listdir('test_data')
    for file in files:
        input_image_path = 'test_data/' + file
        if '+' not in file and os.path.isfile(input_image_path):
            start = time.time()
            predictions, rgb, img_normalized = xp.load_and_predict_image(input_image_path)
            elapsed = time.time() - start
            print('Time elapsed: %.02f sec' % elapsed)

            tmp = rgb * 0
            for c in range(3):
                tmp[:, :, c] = img_normalized
            combined = np.concatenate((tmp, rgb), axis=1)

            prob = predictions['abnormal_lungs']
            if not to_plot:
                out_path = '%s+vgg16-abnorm%.02f.png' % (input_image_path, prob)
                io.imsave(out_path, combined)

                out_path = '%s+vgg16-abnorm%.02f.json' % (input_image_path, prob)
                with open(out_path, 'w') as f:
                    json.dump(predictions, f, indent=2)
            else:
                plt.imshow(combined)
                title = '%s+vgg16 (%.2f sec)\n' % (input_image_path, elapsed)
                title += pred2str(predictions)
                plt.title(title)
                plt.pause(1.0)


if __name__ == '__main__':
    main()
