import os
import json
import keras
import numpy as np
import pandas as pd
from random import shuffle
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


def process_item(data_dir, data_shape, item_class, item_path, x, y, i, df):
    if i % 1000 == 0:
        print('%i / %i' % (i, df.shape[0]))

    path = os.path.join(data_dir, item_path)
    if os.path.isfile(path):
        img = img_as_float(io.imread(path))
        if img.shape != data_shape:
            img = transform.resize(img, data_shape)
        img = (img * 255).astype(np.uint8)

        x.append(np.expand_dims(img, -1))
        y.append(np.array([item_class]))


def load_data(data_dir, data_shape, shuffle_lines=True):
    print('Loading data from ' + data_dir)

    x_train_val = []
    y_train_val = []
    for filename in ['train.txt', 'val.txt']:
        x = []
        y = []

        df = pd.read_csv(os.path.join(data_dir, filename), sep=' ', header=None)

        idx = list(range(df.shape[0]))
        if shuffle_lines:
            print('Shuffling lines')
            shuffle(idx)

            for i, row_idx in enumerate(idx):
                item_path = df[0][row_idx]
                item_class = df[1][row_idx]
                process_item(data_dir, data_shape, item_class, item_path, x, y, i, df)
        else:
            for i, row in df.iterrows():
                item_path = row[0]
                item_class = row[1]
                process_item(data_dir, data_shape, item_class, item_path, x, y, i, df)

        x = np.array(x).astype(np.float32) / 255.
        x -= 0.5
        x_train_val.append(x)
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

    model = model_type(weights=None, include_top=True, input_shape=(final_image_sz, final_image_sz, 1),
                       classes=num_classes)

    json_path = os.path.join(data_dir, 'config.json')
    print('Reading config from ' + json_path)
    with open(json_path, 'r') as f:
        config = json.load(f)

    pattern = 'models/%s_Sz%i_%s_%s_Ep%i_Lr%.1e*.hdf5'
    pattern = pattern % (config['dataset'], image_sz, model_type.__name__, optimizer.__class__.__name__,
                         epochs, learning_rate)

    print('\n### Running training for ' + pattern + '\n')

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    train_gen = ModifiedDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, rescale=1.,
                                      zoom_range=0.2, fill_mode='nearest', cval=0, crop_to=crop_to)

    val_gen = ModifiedDataGenerator(rescale=1., crop_to=crop_to)

    tensor_board = keras.callbacks.TensorBoard(log_dir=pattern.replace('models/', 'graphs/').replace('*.hdf5', ''),
                                               histogram_freq=0, write_graph=True, write_images=True)

    callbacks = [tensor_board]

    if optimizer.__class__.__name__ == 'SGD':
        def schedule(epoch):
            if epoch < epochs // 3:
                return learning_rate
            if epoch < 2 * epochs // 3:
                return learning_rate * 0.1
            return learning_rate * 0.01

        callbacks.append(keras.callbacks.LearningRateScheduler(schedule=schedule))

    model.fit_generator(train_gen.flow(x_train, y_train, batch_size),
                        steps_per_epoch=(x_train.shape[0] + batch_size - 1) // batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
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
        optimizer = keras.optimizers.rmsprop(lr=learning_rate, decay=1.e-6)
    elif optimizer == 'SGD':
        optimizer = keras.optimizers.sgd(lr=learning_rate, decay=1.e-6, nesterov=True, momentum=0.9)
    elif optimizer == 'Adam':
        optimizer = keras.optimizers.adam(lr=learning_rate, decay=1.e-6)

    return batch_size, data_dir, epochs, image_sz, learning_rate, model_type, num_classes, optimizer, crop_to


def main():
    batch_size, data_dir, epochs, image_sz, learning_rate, model_type, num_classes, optimizer, crop_to = parse_args()

    train_model(batch_size, data_dir, epochs, image_sz, learning_rate, model_type, num_classes, optimizer, crop_to)


if __name__ == '__main__':
    # os.sys.argv = 'train_model.py 32 /home/skliff13/work/PTD_Xray/datasets/tuberculosis/v2.3 3 299 0.0001 InceptionV3 2 RMSprop -1'.split(' ')

    main()
