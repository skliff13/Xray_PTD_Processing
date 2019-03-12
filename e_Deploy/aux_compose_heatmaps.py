import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from skimage import io

from imutils import imresize


def fit_to(im, sz):
    result = np.ones((sz, sz, 3), dtype=np.uint8) * 255

    factor = min(sz / im.shape[0], sz / im.shape[1])
    new_shape = (int(im.shape[0] * factor + 0.5), int(im.shape[1] * factor + 0.5))
    sub = imresize(im, new_shape)

    r0 = (sz - sub.shape[0]) // 2
    r1 = (sz - sub.shape[1]) // 2
    result[r0:r0 + sub.shape[0], r1:r1 + sub.shape[1], :] = sub

    result[0, :, :] = 255
    result[-1, :, :] = 255
    result[:, 0, :] = 255
    result[:, -1, :] = 255

    return result


im_dir = 'test_data/val'

cases = ['emphysema_02.png', 'fibrosis_02.png', 'focal_shadows_04.png', 'pneumonia_03.png', 'tuberculosis_01.png']
# cases = ['emphysema_02.png', 'fibrosis_02.png', 'pneumonia_03.png', 'tuberculosis_01.png']
nets = ['vgg16_1', 'vgg19_1', 'inceptionv3_1']

sz = 512
case_is_col = 1

rows = []
for case in cases:
    row = []
    for net in nets:
        pattern = os.path.join(im_dir, net, case + '+*.png')

        files = glob(pattern)
        im_path = files[0]

        im = io.imread(im_path)

        orig = fit_to(im[:, :im.shape[1] // 2, :], sz)
        hm = fit_to(im[:, im.shape[1] // 2:, :], sz)

        if not row:
            row.append(orig)
        row.append(hm)

    row = np.concatenate(row, axis=((case_is_col + 1) % 2))
    rows.append(row)

big = np.concatenate(rows, axis=(case_is_col % 2))

# io.imsave('composed_heatmaps.png', big)
plt.imshow(big)
plt.title(str(cases))
plt.show()

