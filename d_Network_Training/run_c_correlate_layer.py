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
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

from load_data import get_train_batches, get_val_batches, load_batch


def save_layer_coefs(batch_size, data_dir, image_sz, layer_model, model_path, layer_name, mode_train):
    data_shape = (image_sz, image_sz)
    if mode_train:
        batches, num_cases = get_train_batches(data_dir, batch_size)
    else:
        batches, num_cases = get_val_batches(data_dir, batch_size)

    # predictions = np.zeros((0, 2), dtype=float)
    y = np.zeros((0, 1), dtype=float)
    averages = None
    paths = []
    for i, batch in enumerate(batches):
        print('Processing batch %i / %i' % (i + 1, len(batches)))

        x_batch, y_batch = load_batch(data_dir, data_shape, batch)

        # predictions_batch = model.predict(x_batch, batch_size=batch_size)

        layer_output = layer_model.predict(x_batch, batch_size=batch_size)
        averages_batch = np.mean(layer_output, axis=(1, 2))

        # predictions = np.concatenate((predictions, predictions_batch), axis=0)
        y = np.concatenate((y, y_batch), axis=0)
        if averages is None:
            averages = averages_batch
        else:
            averages = np.concatenate((averages, averages_batch), axis=0)

        for item_path, _ in batch:
            paths.append(item_path)

    # fpr, tpr, _ = roc_curve(y[:, 0].ravel(), predictions[:, 1].ravel())
    # print('Predictions AUC: %f' % auc(fpr, tpr))

    reg = LinearRegression().fit(averages, y[:, 0])
    coefs = np.append(reg.coef_, [reg.intercept_])

    act_pred = np.matmul(averages, coefs[:-1])
    fpr, tpr, _ = roc_curve(y[:, 0].ravel(), act_pred.ravel())
    print('Linear model with activations AUC: %f' % auc(fpr, tpr))

    if mode_train:
        out_path = model_path[:-5] + '_' + layer_name + '_coefs.txt'
        print('Saving coefficients to ' + out_path)
        np.savetxt(out_path, coefs)

        rs = []
        for j in range(averages.shape[1]):
            rs.append(pearsonr(averages[:, j], y[:, 0])[0])
        print('Saving correlations to ' + out_path)
        out_path = model_path[:-5] + '_' + layer_name + '_corrs.txt'
        np.savetxt(out_path, np.array(rs))

    model_filename = os.path.split(model_path)[-1]
    out_dir = os.path.join(data_dir, 'predictions', model_filename[:-5])
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(averages)
    df['path'] = pd.Series(paths, index=df.index)
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    if mode_train:
        out_path = os.path.join(out_dir, layer_name + '_averages_train.txt')
    else:
        out_path = os.path.join(out_dir, layer_name + '_averages_val.txt')
    print('Saving averages to ' + out_path)
    df.to_csv(out_path, index=None)


def main():
    num_classes = 2
    image_sz = 224
    model_type = VGG16
    batch_size = 16
    # data_dir = '/home/skliff13/work/PTD_Xray/datasets/tuberculosis/v2.3'
    data_dir = '/home/skliff13/work/PTD_Xray/datasets/abnormal_lungs/v2.0'
    # model_path = 'models/abnormal_lungs_v2.0_Sz299_InceptionV3_RMSprop_Ep50_Lr1.0e-04_Auc0.880.hdf5'
    # model_path = 'models/_old/model_Sz299_InceptionV3_RMSprop_Ep300_Lr1.0e-04_Auc0.864.hdf5'
    # layer_name = 'mixed10'
    model_path = 'models/abnormal_lungs_v2.0_Sz224_VGG16_Adam_Ep30_Lr1.0e-05_Auc0.851.hdf5'
    layer_name = 'block5_conv3'

    model = model_type(weights=None, include_top=True, input_shape=(image_sz, image_sz, 1), classes=num_classes)

    print('Loading model ' + model_path)
    model.load_weights(model_path)
    layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    for mode_train in [True, False]:
        save_layer_coefs(batch_size, data_dir, image_sz, layer_model, model_path, layer_name, mode_train)


if __name__ == '__main__':
    main()
