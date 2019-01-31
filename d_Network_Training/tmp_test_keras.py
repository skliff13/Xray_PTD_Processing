import os
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


def load_data(data_dir, data_shape):
    print('Loading data from ' + data_dir)

    x_train_val = []
    y_train_val = []
    for filename in ['train.txt', 'val.txt']:
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

        x_train_val.append(np.array(x))
        y_train_val.append(np.array(y))

    print('train_data:', x_train_val[0].shape, y_train_val[0].shape)
    print('val_data:', x_train_val[1].shape, y_train_val[1].shape)

    return (x_train_val[0], y_train_val[0]), (x_train_val[1], y_train_val[1])


def train_model(batch_size, data_dir, epochs, image_sz, learning_rate, model_type, num_classes, optimizer):
    data_shape = (image_sz, image_sz)
    (x_train, y_train), (x_val, y_val) = load_data(data_dir, data_shape)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    model = model_type(weights=None, include_top=True, input_shape=(image_sz, image_sz, 1), classes=num_classes)

    pattern = 'model_Sz%i_%s_%s_Ep%i_Lr%.1e*.hdf5'
    pattern = pattern % (image_sz, model_type.__name__, optimizer.__name__, epochs, learning_rate)

    files = glob(pattern)
    if not files:
        print('\n### Running training for ' + pattern + '\n')

        opt = optimizer(lr=learning_rate, decay=0.0e-6)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        train_gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, rescale=1.,
                                       zoom_range=0.2, fill_mode='nearest', cval=0)

        val_gen = ImageDataGenerator(rescale=1.)

        model.fit_generator(train_gen.flow(x_train, y_train, batch_size),
                            steps_per_epoch=(x_train.shape[0] + batch_size - 1) // batch_size,
                            epochs=epochs,
                            validation_data=val_gen.flow(x_val, y_val),
                            validation_steps=(x_val.shape[0] + batch_size - 1) // batch_size)

        predictions = model.predict(x_val, batch_size=batch_size)
        fpr, tpr, _ = roc_curve(y_val[:, 1].ravel(), predictions[:, 1].ravel())
        roc_auc = auc(fpr, tpr)

        model_filename = pattern.replace('*', '_Auc%.3f' % roc_auc)

        model.save_weights(model_filename)


def main():
    num_classes = 2
    image_sz = 224
    model_type = InceptionV3
    data_dir = '/home/skliff13/work/PTD_Xray/datasets/tuberculosis/v2.2'
    epochs = 300
    batch_size = 32
    learning_rate = 1e-4
    optimizer = keras.optimizers.rmsprop

    # for optimizer in [keras.optimizers.sgd, keras.optimizers.adam, keras.optimizers.rmsprop]:
    for model_type in [VGG16, VGG19, InceptionV3]:
        for image_sz in [224, 256, 299]:
            train_model(batch_size, data_dir, epochs, image_sz, learning_rate, model_type, num_classes, optimizer)


main()