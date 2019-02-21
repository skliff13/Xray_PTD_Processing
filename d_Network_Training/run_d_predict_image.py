import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_SUB_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
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


def load_prepared_image(path, data_shape):
    img = img_as_float(io.imread(path))
    img = transform.resize(img, data_shape)
    img -= 0.5
    img = np.expand_dims(img, axis=-1)
    return img


def get_random_val(data_dir):
    file_path = os.path.join(data_dir, 'val.txt')

    df = pd.read_csv(file_path, sep=' ', header=None)
    r = randint(0, df.shape[0] - 1)

    img_path = os.path.join(data_dir, df[0][r])
    img_class = int(df[1][r])

    return img_path, img_class


def main():
    num_classes = 2
    image_sz = 256
    model_type = VGG16
    data_dir = '/home/skliff13/work/PTD_Xray/datasets/tuberculosis/v2.2'
    batch_size = 16
    model_path = 'models/_old/model_Sz256_VGG16_RMSprop_Ep300_Lr1.0e-04_Auc0.818.hdf5'
    layer_name = 'block5_conv3'
    # img_class = -1
    # img_path = '/home/skliff13/work/PTD_Xray/datasets/tuberculosis/v2.2/img/997802_1136_1_32951.dcm.png_aug000.png'
    # img_path = '/home/skliff13/work/PTD_Xray/datasets/tuberculosis/v2.2/img/005878_2278_5_271242.dcm.png_aug000.png'

    model = model_type(weights=None, include_top=True, input_shape=(image_sz, image_sz, 1), classes=num_classes)
    print('Loading model ' + model_path)
    model.load_weights(model_path)

    coefs_path = model_path[:-5] + '_' + layer_name + '_coefs.txt'
    print('Loading coefs ' + coefs_path)
    coefs = np.loadtxt(coefs_path)

    while True:
        img_path, img_class = get_random_val(data_dir)

        print('Reading image ' + img_path)
        x = load_prepared_image(img_path, (image_sz, image_sz))
        x = x[np.newaxis, ...]

        print('Predicting values')
        prediction = model.predict(x, batch_size=batch_size)

        print('Evaluating activations')
        layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        layer_output = layer_model.predict(x, batch_size=batch_size)

        heatmap = np.matmul(layer_output, coefs[:-1])[0, ...] + coefs[-1]
        heatmap = transform.resize(heatmap, (image_sz, image_sz))
        heatmap[heatmap < 0] = 0
        heatmap[heatmap > 1] = 1

        both = np.concatenate((x[0, :, :, 0] + 0.5, heatmap), axis=1)
        plt.imshow(both, cmap='gray')
        plt.colorbar()
        plt.title('Class = %i, Prediction: %i%%' % (img_class, int(round(100 * prediction[0, 1]))))
        plt.show()


if __name__ == '__main__':
    main()
