import os
# if __name__ == '__main__':
#     os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.visible_device_list = "1"
set_session(tf.Session(config=config))
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import roc_curve, auc
from skimage import io, img_as_float, transform
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
from keras_applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras_applications.resnet50 import ResNet50
from inception_v1 import InceptionV1


def parse_model_type(model_type):
    if model_type == 'VGG16':
        return VGG16
    if model_type == 'VGG19':
        return VGG19
    if model_type == 'InceptionV1':
        return InceptionV1
    if model_type == 'InceptionV3':
        return InceptionV3
    if model_type == 'InceptionResNetV2':
        return InceptionResNetV2
    if model_type == 'ResNet50':
        return ResNet50


def importance_as_weights_sum(weights):
    w = weights[0]
    # b = weights[1]

    w_shape = w.shape
    prod = 1
    for i in range(len(w_shape) - 1):
        prod *= w_shape[i]
    x = w.reshape((prod, w_shape[-1])).copy()
    # x = np.append(x, b.reshape(1, w_shape[-1]), axis=0)

    x = np.sum(np.abs(x), axis=1)
    return x


def importance_as_dist(weights):
    w = weights[0]
    # b = weights[1]

    w_shape = w.shape
    prod = 1
    for i in range(len(w_shape) - 1):
        prod *= w_shape[i]
    x = w.reshape((prod, w_shape[-1])).copy()
    # x = np.append(x, b.reshape(1, w_shape[-1]), axis=0)

    dm = distance_matrix(x, x).flatten()
    return dm


def importance_as_binned_copies(weights):
    w = weights[0]
    # b = weights[1]

    w_shape = w.shape
    prod = 1
    for i in range(len(w_shape) - 1):
        prod *= w_shape[i]
    x = w.reshape((prod, w_shape[-1])).copy()
    # x = np.append(x, b.reshape(1, w_shape[-1]), axis=0)

    x /= np.tile(np.sum(x**2, axis=0), (x.shape[0], 1))
    shift = np.max(np.abs(x)) * 1.01
    x += shift

    uniques = []
    for nbins in [32, 16, 12, 8, 6, 4, 2]:
        b = np.floor(x / 2. / shift * nbins).astype(int)

        l = []
        for c in range(b.shape[1]):
            l.append(tuple(b[:, c]))

        s = set(l)
        u = float(len(s)) / len(l)
        uniques.append(u)

    return np.array(uniques)


def importance_as_pca(weights):
    w = weights[0]
    # b = weights[1]

    w_shape = w.shape
    prod = 1
    for i in range(len(w_shape) - 1):
        prod *= w_shape[i]
    x = w.reshape((prod, w_shape[-1])).copy()
    # x = np.append(x, b.reshape(1, w_shape[-1]), axis=0)

    pca = PCA()
    pca.fit(np.transpose(x))
    ev = pca.explained_variance_
    cs = np.cumsum(ev) / np.sum(ev)
    n80 = np.argwhere(cs > 0.80)[0]
    n90 = np.argwhere(cs > 0.90)[0]
    n95 = np.argwhere(cs > 0.95)[0]
    n99 = np.argwhere(cs > 0.99)[0]

    print(n80, '/', n90, '/', n95, '/', n99, '(', ev.shape[0], ')')

    result = np.zeros((w_shape[-1], ), dtype=float)
    result[0:ev.shape[0]] = ev

    return result


def analyze_layer(weights, name):
    x = importance_as_pca(weights)

    f = list(x / np.max(x))
    f.sort(reverse=True)

    xx = np.linspace(0, 1, len(f))
    plt.plot(xx, f, label=name)


def main():
    # model_filename = 'models/_old/model_Sz256_VGG16_RMSprop_Ep300_Lr1.0e-04_Auc0.818.hdf5'
    # model_filename = 'models/_old/model_Sz224_VGG19_RMSprop_Ep300_Lr1.0e-04_Auc0.793.hdf5'
    # model_filename = 'models/_old/model_Sz299_InceptionV3_RMSprop_Ep300_Lr1.0e-04_Auc0.864.hdf5'
    model_filename = 'models/abnormal_lungs_v2.0_Sz224_VGG16_Adam_Ep30_Lr1.0e-05_Auc0.851.hdf5'
    # model_filename = 'models/abnormal_lungs_v2.0_Sz224_VGG19_Adam_Ep30_Lr1.0e-05_Auc0.847.hdf5'
    num_classes = 2

    parts = os.path.split(model_filename)[1].split('_')
    image_sz = int(parts[-6][2:])
    model_type = os.path.split(model_filename)[1].split('_')[-5]

    model_type = parse_model_type(model_type)
    model = model_type(weights=None, include_top=True, input_shape=(image_sz, image_sz, 1), classes=num_classes)

    plt.figure()

    model_filename = os.path.join(os.path.split(__file__)[0], model_filename)
    if os.path.isfile(model_filename):
        print('Loading model from ' + model_filename)
        model.load_weights(model_filename)

        for layer in model.layers:
            weights = layer.get_weights()

            print(layer.name, layer.output_shape)

            # if len(weights) == 2 and len(weights[0].shape) == 4:
            #     analyze_layer(weights, layer.name)

    plt.legend()
    plt.show()


main()