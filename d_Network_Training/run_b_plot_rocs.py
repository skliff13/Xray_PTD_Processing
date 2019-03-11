import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))
import keras
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import roc_curve, auc
from skimage import io, img_as_float, transform
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

from load_data import load_val_data


def evaluate_model(batch_size, data_dir, image_sz, model_type, num_classes, weights_path):
    if not os.path.isfile(weights_path):
        print('\n### File not found: ' + weights_path + '\n')

        return None, None, None

    model_filename = os.path.split(weights_path)[-1]
    pred_dir = os.path.join(data_dir, 'predictions', model_filename[:-5])
    pathlib.Path(pred_dir).mkdir(parents=True, exist_ok=True)

    pred_path = os.path.join(pred_dir, 'pred_val.txt')
    if os.path.isfile(pred_path):
        print('Loading cached predictions from ' + pred_path)
        predictions = pd.read_csv(pred_path, header=None).get_values()
    else:
        data_shape = (image_sz, image_sz)
        (x_val, _) = load_val_data(data_dir, data_shape)

        model = model_type(weights=None, include_top=True, input_shape=(image_sz, image_sz, 1), classes=num_classes)

        print('Loading model ' + weights_path)
        model.load_weights(weights_path)

        print('Predicting values')
        predictions = model.predict(x_val, batch_size=batch_size)

        print('Saving predictions to ' + pred_path)
        df = pd.DataFrame(data=predictions)
        df.to_csv(pred_path, header=None, index=None)

    return predictions, weights_path


def make_combined(d):
    if len(d) < 2:
        return

    sum_pred = None
    for model_file in d:
        if sum_pred is None:
            sum_pred = d[model_file].copy()
        else:
            sum_pred += d[model_file]
    d['combined'] = sum_pred


def main():
    num_classes = 2
    data_dir = '/home/skliff13/work/PTD_Xray/datasets/abnormal_lungs/v2.0'
    batch_size = 16

    tests = list()
    tests.append([VGG16, 224, 'models/abnormal_lungs_v2.0_Sz224_VGG16_Adam_Ep30_Lr1.0e-05_Auc0.851.hdf5'])
    tests.append([VGG19, 224, 'models/abnormal_lungs_v2.0_Sz224_VGG19_Adam_Ep30_Lr1.0e-05_Auc0.847.hdf5'])
    tests.append([InceptionV3, 299, 'models/abnormal_lungs_v2.0_Sz299_InceptionV3_RMSprop_Ep50_Lr1.0e-04_Auc0.880.hdf5'])

    df = pd.read_csv(os.path.join(data_dir, 'val.txt'), header=None, sep=' ')
    y_val = df[1].get_values()
    y_val = keras.utils.to_categorical(y_val, num_classes)

    d = {}
    for model_type, image_sz, weights_path in tests:
        pred, model_file = evaluate_model(batch_size, data_dir, image_sz, model_type, num_classes, weights_path)
        if pred is not None:
            d[model_type.__name__] = pred

    # make_combined(d)

    lw = 2
    # plt.figure(figsize=(4.5, 4.5), dpi=600)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], color='lime', lw=lw, linestyle='--', label='baseline')

    line_styles = [':', '-.', '-']
    counter = 0

    for model_file in d:
        pred = d[model_file]
        fpr, tpr, _ = roc_curve(y_val[:, 1].ravel(), pred[:, 1].ravel())
        roc_auc = auc(fpr, tpr)

        ls = line_styles[counter]
        counter = (counter + 1) % len(line_styles)

        plt.plot(fpr, tpr, lw=lw, label=('%s (AUC %0.3f)' % (model_file, roc_auc)), linestyle=ls)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    # plt.title('ROC curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', prop={'size': 10})
    # plt.savefig('fig.png')
    plt.show()


if __name__ == '__main__':
    main()
