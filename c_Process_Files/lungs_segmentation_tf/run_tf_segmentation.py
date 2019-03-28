import os
import sys
import numpy as np
from glob import glob
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.visible_device_list = "1"
set_session(tf.Session(config=config))
from keras.models import load_model
from skimage import morphology, io, img_as_float


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

    sz = 256
    im_shape = (sz, sz)

    model_path = os.path.join(os.path.split(__file__)[0], 'trained_model.hdf5')
    print('Loading model ' + model_path)
    UNet = load_model(model_path)

    print('Scanning ' + path + '/*.png')
    for file_path in glob(path + '/*.png'):
        if not file_path.endswith('-mask.png'):
            out_path = file_path[:-4] + '-mask.png'

            if os.path.isfile(out_path):
                print('Skipping ' + file_path)
            else:
                print('Processing ' + file_path)

                img = img_as_float(io.imread(file_path))

                img -= img.mean()
                img /= img.std()

                pred = img * 0

                batch_size = int(round(img.shape[1] / img.shape[0]))
                for i in range(batch_size):
                    x = img[:, i * sz:(i + 1) * sz]
                    x = np.expand_dims(x, axis=0)
                    x = np.expand_dims(x, axis=-1)

                    pr = UNet.predict(x)[..., 0].reshape(im_shape)
                    pr = pr > 0.5
                    pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))

                    pred[:, i * sz:(i + 1) * sz] = pr

                print('Saving to "%s"' % out_path)
                io.imsave(out_path, (pred.astype(float) * 255).astype(np.uint8))
