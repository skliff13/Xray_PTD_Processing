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
import pandas as pd
from keras.models import Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, auc
from scipy.stats.stats import pearsonr
import json
import numpy as np
from keras_applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras_applications.resnet50 import ResNet50

from load_data import get_train_batches, get_val_batches, load_batch


def parse_model_type(model_type_name):
    if model_type_name == 'VGG16':
        return VGG16
    elif model_type_name == 'VGG19':
        return VGG19
    elif model_type_name == 'InceptionV3':
        return InceptionV3
    elif model_type_name == 'InceptionResNetV2':
        return InceptionResNetV2
    elif model_type_name == 'ResNet50':
        return ResNet50
    else:
        print('Unknown net model: ' + model_type_name)
        exit(1)
        return None


def save_layer_coefs(batch_size, data_dir, image_sz, map_layer_model, desc_layer_model, weights_path, map_layer_name,
                     desc_layer_name, mode_train):
    data_shape = (image_sz, image_sz)
    if mode_train:
        batches, _ = get_train_batches(data_dir, batch_size)
    else:
        batches, _ = get_val_batches(data_dir, batch_size)

    y = np.zeros((0, 1), dtype=float)
    averages = None
    descs = None
    paths = []
    for i, batch in enumerate(batches):
        print('Processing batch %i / %i' % (i + 1, len(batches)))

        x_batch, y_batch = load_batch(data_dir, data_shape, batch)

        map_layer_output = map_layer_model.predict(x_batch, batch_size=batch_size)
        averages_batch = np.mean(map_layer_output, axis=(1, 2))

        if desc_layer_name == map_layer_name:
            desc_batch = averages_batch
        else:
            desc_batch = desc_layer_model.predict(x_batch, batch_size=batch_size)
            if len(desc_batch.shape) == 4:
                desc_batch = np.mean(desc_batch, axis=(1, 2))

        y = np.concatenate((y, y_batch), axis=0)

        if averages is None:
            averages = averages_batch
        else:
            averages = np.concatenate((averages, averages_batch), axis=0)

        if descs is None:
            descs = desc_batch
        else:
            descs = np.concatenate((descs, desc_batch), axis=0)

        for item_path, _ in batch:
            paths.append(item_path)

    # assess_auc(descs, 'desc', y)

    if mode_train:
        coefs = assess_auc(averages, 'map', y)

        out_path = weights_path[:-5] + '_' + map_layer_name + '_train_coefs.txt'
        print('Saving coefficients to ' + out_path)
        np.savetxt(out_path, coefs)

        rs = []
        for j in range(averages.shape[1]):
            rs.append(pearsonr(averages[:, j], y[:, 0])[0])
        print('Saving correlations to ' + out_path)
        out_path = weights_path[:-5] + '_' + map_layer_name + '_train_corrs.txt'
        np.savetxt(out_path, np.array(rs))

    model_filename = os.path.split(weights_path)[-1]
    out_dir = os.path.join(data_dir, 'predictions', model_filename[:-5])
    os.makedirs(out_dir, exist_ok=True)

    save_layer_outputs(descs, desc_layer_name, mode_train, out_dir, paths)


def assess_auc(x, kind, y):
    print('Training Logit classifier')
    reg = LinearRegression().fit(x, y[:, 0])
    coefs = np.append(reg.coef_, [reg.intercept_])
    act_pred = np.matmul(x, coefs[:-1])
    fpr, tpr, _ = roc_curve(y[:, 0].ravel(), act_pred.ravel())
    print('Logistic Regression with %s layer mean activations: AUC %f' % (kind, auc(fpr, tpr)))
    return coefs


def save_layer_outputs(outputs, layer_name, mode_train, out_dir, paths):
    df = pd.DataFrame(outputs)
    df['path'] = pd.Series(paths, index=df.index)
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    if mode_train:
        out_path = os.path.join(out_dir, layer_name + '_descs_train.txt')
    else:
        out_path = os.path.join(out_dir, layer_name + '_descs_val.txt')
    print('Saving averages to ' + out_path)
    df.to_csv(out_path, index=None)


def main():
    num_classes = 2
    batch_size = 16

    with open('setup_vgg19_1.json', 'r') as f:
        d = json.load(f)

    image_sz = d['image_sz']
    model_type = parse_model_type(d['model_type'])
    data_dir = d['data_dir']
    weights_path = d['weights_path']
    map_layer_name = d['map_layer_name']
    desc_layer_name = d['desc_layer_name']

    model = model_type(weights=None, include_top=True, input_shape=(image_sz, image_sz, 1), classes=num_classes)

    print('Loading weigths ' + weights_path)
    model.load_weights(weights_path)
    map_layer_model = Model(inputs=model.input, outputs=model.get_layer(map_layer_name).output)
    if desc_layer_name != map_layer_name:
        desc_layer_model = Model(inputs=model.input, outputs=model.get_layer(desc_layer_name).output)
    else:
        desc_layer_model = None

    for mode_train in [True, False]:
        save_layer_coefs(batch_size, data_dir, image_sz, map_layer_model, desc_layer_model, weights_path,
                         map_layer_name, desc_layer_name, mode_train)


if __name__ == '__main__':
    main()
