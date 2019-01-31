import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_SUB_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import roc_curve, auc
from skimage import io, img_as_float, transform
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator


def load_val_data(data_dir, data_shape):
    print('Loading validation data from ' + data_dir)

    x_val = []
    y_val = []

    filename = 'val.txt'
    x = []
    y = []

    df = pd.read_csv(os.path.join(data_dir, filename), sep=' ', header=None)
    for i, row in df.iterrows():
        path = os.path.join(data_dir, row[0])
        if os.path.isfile(path):
            img = img_as_float(io.imread(path))
            img = transform.resize(img, data_shape)
            img -= 0.5

            x.append(np.expand_dims(img, -1))
            y.append(np.array([row[1]]))

    x_val.append(np.array(x))
    y_val.append(np.array(y))

    print('val_data:', x_val[0].shape, y_val[0].shape)

    return (x_val[0], y_val[0])


def evaluate_model(batch_size, data_dir, epochs, image_sz, learning_rate, model_type, num_classes, optimizer):
    data_shape = (image_sz, image_sz)
    (x_val, y_val) = load_val_data(data_dir, data_shape)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    model = model_type(weights=None, include_top=True, input_shape=(image_sz, image_sz, 1), classes=num_classes)

    pattern = 'model_Sz%i_%s_%s_Ep%i_Lr%.1e*.hdf5'
    pattern = pattern % (image_sz, model_type.__name__, optimizer.__name__, epochs, learning_rate)

    files = glob(pattern)
    if not files:
        print('\n### File not found: ' + pattern + '\n')

        return None, None, None
    else:
        print('Loading model ' + files[0])
        model.load_weights(files[0])

        print('Predicting values')
        predictions = model.predict(x_val, batch_size=batch_size)

        return predictions, files[0], y_val


def main():
    num_classes = 2
    image_sz = 224
    model_type = InceptionV3
    data_dir = '/home/skliff13/work/PTD_Xray/datasets/tuberculosis/v2.2'
    epochs = 300
    batch_size = 32
    learning_rate = 1e-4
    optimizer = keras.optimizers.rmsprop

    d = {}
    for model_type in [InceptionV3]:
        for image_sz in [256, 299]:
            pred, model, y_val = evaluate_model(batch_size, data_dir, epochs, image_sz, learning_rate,
                                         model_type, num_classes, optimizer)
            if pred is not None:
                d[model] = pred

    sum_pred = None
    for model in d:
        if sum_pred is None:
            sum_pred = d[model]
        else:
            sum_pred += d[model]
    d['combined'] = sum_pred

    lw = 2
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], color='green', lw=lw, linestyle='--', label='baseline')

    for model in d:
        pred = d[model]
        fpr, tpr, _ = roc_curve(y_val[:, 1].ravel(), pred[:, 1].ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, label=('%s (AUC %0.3f)' % (model, roc_auc)))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()


main()