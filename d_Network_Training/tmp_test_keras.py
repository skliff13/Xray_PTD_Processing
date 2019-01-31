import os
import keras
import numpy as np
import pandas as pd
from skimage import io, img_as_float, transform
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
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


def main():
    num_classes = 2
    sz = 224
    data_shape = (sz, sz)
    data_dir = '/home/skliff13/work/PTD_Xray/datasets/tuberculosis/v2.2'

    (x_train, y_train), (x_val, y_val) = load_data(data_dir, data_shape)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    print(y_val.shape)

    # model = InceptionV3(input_shape=(256, 256, 1), include_top=False, weights=None)
    model = VGG16(weights=None, include_top=True, input_shape=(sz, sz, 1), classes=num_classes)

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=8,
              epochs=3,
              validation_data=(x_val, y_val),
              shuffle=True)

    # train_gen = ImageDataGenerator(rotation_range=10,
    #                                width_shift_range=0.1,
    #                                height_shift_range=0.1,
    #                                rescale=1.,
    #                                zoom_range=0.2,
    #                                fill_mode='nearest',
    #                                cval=0)
    #
    # val_gen = ImageDataGenerator(rescale=1.)
    #
    # batch_size = 8
    # model.fit_generator(train_gen.flow(x_train, y_train, batch_size),
    #                    steps_per_epoch=(x_train.shape[0] + batch_size - 1) // batch_size,
    #                    epochs=3,
    #                    validation_data=val_gen.flow(x_val, y_val),
    #                    validation_steps=(x_val.shape[0] + batch_size - 1) // batch_size)


main()