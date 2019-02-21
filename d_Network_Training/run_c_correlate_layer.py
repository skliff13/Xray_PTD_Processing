import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_SUB_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import keras
from keras.models import Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, auc
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

from run_b_plot_rocs import load_val_data


def save_layer_coefs(batch_size, data_dir, image_sz, model_type, num_classes, model_path, layer_name):
    data_shape = (image_sz, image_sz)
    (x_val, y_val) = load_val_data(data_dir, data_shape)

    model = model_type(weights=None, include_top=True, input_shape=(image_sz, image_sz, 1), classes=num_classes)

    print('Loading model ' + model_path)
    model.load_weights(model_path)

    print('Predicting values')
    predictions = model.predict(x_val, batch_size=batch_size)

    fpr, tpr, _ = roc_curve(y_val[:, 0].ravel(), predictions[:, 1].ravel())
    print('Predictions AUC: %f' % auc(fpr, tpr))
    exit(13)

    print('Evaluating activations')
    layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    layer_output = layer_model.predict(x_val, batch_size=batch_size)
    averages = np.mean(layer_output, axis=(1, 2))

    reg = LinearRegression().fit(averages, predictions[:, 1])
    coefs = np.append(reg.coef_, [reg.intercept_])

    act_pred = np.matmul(averages, coefs[:-1])
    fpr, tpr, _ = roc_curve(y_val[:, 1].ravel(), act_pred.ravel())
    print('Linear model with activations AUC: %f' % auc(fpr, tpr))

    out_path = model_path[:-5] + '_' + layer_name + '_coefs.txt'
    np.savetxt(out_path, coefs)


def main():
    num_classes = 2
    image_sz = 256
    model_type = VGG16
    data_dir = '/home/skliff13/work/PTD_Xray/datasets/tuberculosis/v2.3'
    batch_size = 32
    # model_path = 'models/_old/model_Sz299_InceptionV3_RMSprop_Ep300_Lr1.0e-04_Auc0.864.hdf5'
    model_path = 'models/_old/model_Sz256_VGG16_RMSprop_Ep300_Lr1.0e-04_Auc0.818.hdf5'
    layer_name = 'block5_conv3'

    save_layer_coefs(batch_size, data_dir, image_sz, model_type, num_classes, model_path, layer_name)


if __name__ == '__main__':
    main()
