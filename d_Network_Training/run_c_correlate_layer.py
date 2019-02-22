import os
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import keras
from keras.models import Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, auc
from scipy.stats.stats import pearsonr
import numpy as np
import pandas as pd
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

from train_model import process_item


def load_train_data(data_dir, data_shape):
    print('Loading data from ' + data_dir)

    filename = 'train.txt'

    x = []
    y = []

    df = pd.read_csv(os.path.join(data_dir, filename), sep=' ', header=None)

    for i, row in df.iterrows():
        item_path = row[0]
        item_class = row[1]
        process_item(data_dir, data_shape, item_class, item_path, x, y, i, df)

    x = np.array(x).astype(np.float32) / 255.
    x -= 0.5
    y = np.array(y)

    print('data:', x.shape, y.shape)

    return x, y


def save_layer_coefs(batch_size, data_dir, image_sz, model_type, num_classes, model_path, layer_name):
    data_shape = (image_sz, image_sz)
    (x_val, y_val) = load_train_data(data_dir, data_shape)

    model = model_type(weights=None, include_top=True, input_shape=(image_sz, image_sz, 1), classes=num_classes)

    print('Loading model ' + model_path)
    model.load_weights(model_path)

    print('Predicting values')
    predictions = model.predict(x_val, batch_size=batch_size)

    fpr, tpr, _ = roc_curve(y_val[:, 0].ravel(), predictions[:, 1].ravel())
    print('Predictions AUC: %f' % auc(fpr, tpr))

    print('Evaluating activations')
    layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    layer_output = layer_model.predict(x_val, batch_size=batch_size)
    averages = np.mean(layer_output, axis=(1, 2))

    reg = LinearRegression().fit(averages, y_val[:, 0])
    coefs = np.append(reg.coef_, [reg.intercept_])

    act_pred = np.matmul(averages, coefs[:-1])
    fpr, tpr, _ = roc_curve(y_val[:, 0].ravel(), act_pred.ravel())
    print('Linear model with activations AUC: %f' % auc(fpr, tpr))

    out_path = model_path[:-5] + '_' + layer_name + '_coefs.txt'
    np.savetxt(out_path, coefs)

    rs = []
    for j in range(averages.shape[1]):
        rs.append(pearsonr(averages[:, j], y_val[:, 0])[0])
    out_path = model_path[:-5] + '_' + layer_name + '_corrs.txt'
    np.savetxt(out_path, np.array(rs))


def main():
    num_classes = 2
    image_sz = 256
    model_type = VGG16
    data_dir = '/home/skliff13/work/PTD_Xray/datasets/tuberculosis/v2.3'
    batch_size = 16
    # model_path = 'models/_old/model_Sz299_InceptionV3_RMSprop_Ep300_Lr1.0e-04_Auc0.864.hdf5'
    # layer_name = 'mixed7'
    model_path = 'models/_old/model_Sz256_VGG16_RMSprop_Ep300_Lr1.0e-04_Auc0.818.hdf5'
    layer_name = 'block5_conv3'

    save_layer_coefs(batch_size, data_dir, image_sz, model_type, num_classes, model_path, layer_name)


if __name__ == '__main__':
    main()
