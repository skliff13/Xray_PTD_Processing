import os
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# config.gpu_options.visible_device_list = "1"
set_session(tf.Session(config=config))
import keras
from keras.models import Model
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, img_as_float, transform
from random import randint
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

from load_data import get_random_val_case, load_prepared_image


def main():
    num_classes = 2
    image_sz = 224
    model_type = VGG16
    # data_dir = '/home/skliff13/work/PTD_Xray/datasets/tuberculosis/v2.3'
    data_dir = '/home/skliff13/work/PTD_Xray/datasets/abnormal_lungs/v2.0'
    batch_size = 1
    # model_path = 'models/_old/model_Sz299_InceptionV3_RMSprop_Ep300_Lr1.0e-04_Auc0.864.hdf5'
    # layer_name = 'mixed10'
    # model_path = 'models/_old/model_Sz256_VGG16_RMSprop_Ep300_Lr1.0e-04_Auc0.818.hdf5'
    model_path = 'models/abnormal_lungs_v2.0_Sz224_VGG16_Adam_Ep30_Lr1.0e-05_Auc0.851.hdf5'
    layer_name = 'block5_conv3'

    model = model_type(weights=None, include_top=True, input_shape=(image_sz, image_sz, 1), classes=num_classes)
    print('Loading model ' + model_path)
    model.load_weights(model_path)

    coefs_path = model_path[:-5] + '_' + layer_name + '_coefs.txt'
    print('Loading coefs ' + coefs_path)
    coefs = np.loadtxt(coefs_path)
    # coefs[coefs < 0] = 0
    # coefs[0:-1] = 1. / coefs.shape[0]
    # coefs[-1] = 0

    corrs_path = model_path[:-5] + '_' + layer_name + '_corrs.txt'
    print('Loading corrs ' + corrs_path)
    corrs = np.loadtxt(corrs_path)
    corrs[np.isnan(corrs)] = 0
    corrs = np.sign(corrs) * np.square(corrs)

    ntests = 16
    for j in range(ntests):
        img_path, img_class = get_random_val_case(data_dir)

        print('Reading image ' + img_path)
        x = load_prepared_image(img_path, (image_sz, image_sz))
        x = x[np.newaxis, ...]

        print('Predicting values')
        prediction = model.predict(x, batch_size=batch_size)

        print('Evaluating activations')
        layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        layer_output = layer_model.predict(x, batch_size=batch_size)

        heatmap1 = np.matmul(layer_output, coefs[:-1])[0, ...] + coefs[-1]
        heatmap2 = np.matmul(layer_output, corrs)[0, ...]

        heatmap1 = transform.resize(heatmap1, (image_sz, image_sz)) / 5
        heatmap1[heatmap1 < 0] = 0
        heatmap1[heatmap1 > 1] = 1

        heatmap2 = transform.resize(heatmap2, (image_sz, image_sz)) * 30
        heatmap2[heatmap2 < 0] = 0
        heatmap2[heatmap2 > 1] = 1

        both = np.concatenate((x[0, :, :, 0] + 0.5, heatmap2), axis=1)
        ax = plt.subplot(np.ceil(ntests**0.5), np.ceil(ntests**0.5), j + 1)
        ax.imshow(both, cmap='gray')
        ax.set_title('Class = %i, Prediction: %i%%' % (img_class, int(round(100 * prediction[0, 1]))))
    plt.show()


if __name__ == '__main__':
    main()
