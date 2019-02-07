import os
import keras
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import roc_curve, auc
from skimage import io, img_as_float, transform
from keras_applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras_applications.resnet50 import ResNet50
# from keras.preprocessing.image import ImageDataGenerator
from inception_v1 import InceptionV1
from data_gen import ModifiedDataGenerator


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


def train_model(batch_size, data_dir, epochs, image_sz, learning_rate, model_type, num_classes, optimizer, crop_to):
    data_shape = (image_sz, image_sz)
    (x_train, y_train), (x_val, y_val) = load_data(data_dir, data_shape)

    if crop_to > 0:
        final_image_sz = crop_to
    else:
        final_image_sz = image_sz

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    model = model_type(weights=None, include_top=True, input_shape=(final_image_sz, final_image_sz, 1), classes=num_classes)

    pattern = 'model_Sz%i_%s_%s_Ep%i_Lr%.1e*.hdf5'
    pattern = pattern % (image_sz, model_type.__name__, optimizer.__name__, epochs, learning_rate)

    print('\n### Running training for ' + pattern + '\n')

    opt = optimizer(lr=learning_rate, decay=0.0e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    train_gen = ModifiedDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, rescale=1.,
                                   zoom_range=0.2, fill_mode='nearest', cval=0, crop_to=crop_to)

    val_gen = ModifiedDataGenerator(rescale=1., crop_to=crop_to)

    model.fit_generator(train_gen.flow(x_train, y_train, batch_size),
                        steps_per_epoch=(x_train.shape[0] + batch_size - 1) // batch_size,
                        epochs=epochs,
                        validation_data=val_gen.flow(x_val, y_val),
                        validation_steps=(x_val.shape[0] + batch_size - 1) // batch_size)

    if crop_to > 0:
        x_val = val_gen.crop_data_bacth(x_val)

    print(x_val.shape)
    predictions = model.predict(x_val, batch_size=batch_size)
    fpr, tpr, _ = roc_curve(y_val[:, 1].ravel(), predictions[:, 1].ravel())
    roc_auc = auc(fpr, tpr)

    model_filename = pattern.replace('*', '_Auc%.3f' % roc_auc)
    print('Saving model to ' + model_filename)
    model.save_weights(model_filename)


def parse_args():
    argv = os.sys.argv
    batch_size = int(argv[1])
    data_dir = argv[2]
    epochs = int(argv[3])
    image_sz = int(argv[4])
    learning_rate = float(argv[5])
    model_type = argv[6]
    num_classes = int(argv[7])
    optimizer = argv[8]
    crop_to = int(argv[9])

    if model_type == 'VGG16':
        model_type = VGG16
    elif model_type == 'VGG19':
        model_type = VGG19
    elif model_type == 'InceptionV1':
        model_type = InceptionV1
    elif model_type == 'InceptionV3':
        model_type = InceptionV3
    elif model_type == 'InceptionResNetV2':
        model_type = InceptionResNetV2
    elif model_type == 'ResNet50':
        model_type = ResNet50

    if optimizer == 'RMSprop':
        optimizer = keras.optimizers.rmsprop
    elif optimizer == 'SGD':
        optimizer = keras.optimizers.sgd
    elif optimizer == 'Adam':
        optimizer = keras.optimizers.adam

    return batch_size, data_dir, epochs, image_sz, learning_rate, model_type, num_classes, optimizer, crop_to


def main():
    batch_size, data_dir, epochs, image_sz, learning_rate, model_type, num_classes, optimizer, crop_to = parse_args()

    train_model(batch_size, data_dir, epochs, image_sz, learning_rate, model_type, num_classes, optimizer, crop_to)


if __name__ == '__main__':
    # os.sys.argv = 'train_model.py 32 /home/skliff13/work/PTD_Xray/datasets/tuberculosis/v2.2 400 256 0.0001 InceptionV1 2 SGD 224'.split(' ')

    main()
