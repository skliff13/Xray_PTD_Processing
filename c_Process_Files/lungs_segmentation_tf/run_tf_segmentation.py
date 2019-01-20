import os
import sys
import numpy as np
from glob import glob
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage import morphology, io, exposure, img_as_float, transform


def load_data(path, im_shape):
    X = []
    filepaths = []
    for filepath in glob(path + '/*.png'):
        if not filepath.endswith('-mask.png'):
            img = img_as_float(io.imread(filepath))
            img = transform.resize(img, im_shape)
            img = exposure.equalize_hist(img)
            img = np.expand_dims(img, -1)

            X.append(img)
            filepaths.append(filepath)

    X = np.array(X)
    X -= X.mean()
    X /= X.std()

    print('### Dataset loaded')
    print('\t{}'.format(path))
    print('\t{}'.format(X.shape))
    print('\tX:{:.1f}-{:.1f}\n'.format(X.min(), X.max()))
    print('\tX.mean = {}, X.std = {}'.format(X.mean(), X.std()))

    return X, filepaths


def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: %s {/path/to/Directory_With_Images}' % os.path.basename(sys.argv[0]))
        sys.exit(1)

    path = sys.argv[1]

    im_shape = (256, 256)
    X, filepaths = load_data(path, im_shape)

    n_test = X.shape[0]
    inp_shape = X[0].shape

    model_name = 'trained_model.hdf5'
    UNet = load_model(model_name)

    test_gen = ImageDataGenerator(rescale=1.)

    for i, X_ in enumerate(X):
        X_ = np.array([X_])
        for xx in test_gen.flow(X_, batch_size=1):
            img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))
            pred = UNet.predict(xx)[..., 0].reshape(inp_shape[:2])

            pr = pred > 0.5
            pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))

            outpath = filepaths[i][:-4] + '-mask.png'
            print('Saving to "%s"' % outpath)
            io.imsave(outpath, (pr.astype(float) * 255).astype(np.uint8))
            break
