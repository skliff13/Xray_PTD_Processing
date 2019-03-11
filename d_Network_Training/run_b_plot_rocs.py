import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.visible_device_list = "1"
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


def evaluate_model(batch_size, data_dir, epochs, image_sz, learning_rate, model_type, num_classes, optimizer):
    json_path = os.path.join(data_dir, 'config.json')
    print('Reading config from ' + json_path)
    with open(json_path, 'r') as f:
        config = json.load(f)

    pattern = 'models/%s_Sz%i_%s_%s_Ep%i_Lr%.1e*.hdf5'
    pattern = pattern % (config['class_of_interest'], image_sz, model_type.__name__, optimizer.__name__, epochs,
                         learning_rate)

    files = glob(pattern)
    if not files:
        print('\n### File not found: ' + pattern + '\n')

        return None, None, None

    weights_path = files[0]

    model_filename = os.path.split(weights_path)[-1]
    pred_dir = os.path.join(data_dir, 'predictions', model_filename[:-5])
    pathlib.Path(pred_dir).mkdir(parents=True, exist_ok=True)

    pred_path = os.path.join(pred_dir, 'pred_val.txt')
    if os.path.isfile(pred_path):
        predictions = pd.read_csv(pred_path, header=None).get_values()
        df = pd.read_csv(os.path.join(data_dir, 'val.txt'), header=None, sep=' ')
        y_val = df[1].get_values()

    else:
        data_shape = (image_sz, image_sz)
        (x_val, y_val) = load_val_data(data_dir, data_shape)

        model = model_type(weights=None, include_top=True, input_shape=(image_sz, image_sz, 1), classes=num_classes)

        print('Loading model ' + weights_path)
        model.load_weights(weights_path)

        print('Predicting values')
        predictions = model.predict(x_val, batch_size=batch_size)

        df = pd.DataFrame(data=predictions)
        df.to_csv(pred_path)

    y_val = keras.utils.to_categorical(y_val, num_classes)
    return predictions, weights_path, y_val


def main():
    num_classes = 2
    # image_sz = 299
    # model_type = InceptionV3
    data_dir = '/home/skliff13/work/PTD_Xray/datasets/tuberculosis/v2.3'
    epochs = 3
    batch_size = 32
    learning_rate = 1e-4
    optimizer = keras.optimizers.rmsprop

    d = {}
    for model_type in [VGG16]:
        for image_sz in [224]:
            pred, model_file, y_val = evaluate_model(batch_size, data_dir, epochs, image_sz, learning_rate,
                                         model_type, num_classes, optimizer)
            if pred is not None:
                d[model_file] = pred

    sum_pred = None
    for model_file in d:
        if sum_pred is None:
            sum_pred = d[model_file].copy()
        else:
            sum_pred += d[model_file]
    d['combined'] = sum_pred

    lw = 2
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], color='lime', lw=lw, linestyle='--', label='baseline')

    for model_file in d:
        pred = d[model_file]
        fpr, tpr, _ = roc_curve(y_val[:, 1].ravel(), pred[:, 1].ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, label=('%s (AUC %0.3f)' % (model_file, roc_auc)))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', prop={'size': 6})
    plt.show()


if __name__ == '__main__':
    main()
