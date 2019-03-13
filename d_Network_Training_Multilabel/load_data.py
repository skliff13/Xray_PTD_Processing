import os
import numpy as np
import pandas as pd
from random import shuffle, randint
from skimage import io, img_as_float, transform


def process_item(data_dir, data_shape, item_classes, item_path, x, y, i, df):
    if i % 1000 == 0:
        print('%i / %i' % (i, df.shape[0]))

    path = os.path.join(data_dir, item_path)
    if os.path.isfile(path):
        img = img_as_float(io.imread(path))
        if img.shape != data_shape:
            img = transform.resize(img, data_shape)
        img = (img * 255).astype(np.uint8)

        x.append(np.expand_dims(img, -1))
        y.append(np.array(item_classes))


def load_data(data_dir, data_shape, num_classes, shuffle_lines=True):
    print('Loading data from ' + data_dir)

    x_train_val = []
    y_train_val = []
    for filename in ['train_%icl.txt' % num_classes, 'val_%icl.txt' % num_classes]:
        x = []
        y = []

        df = pd.read_csv(os.path.join(data_dir, filename), sep=' ', header=None)

        idx = list(df.index)
        if shuffle_lines:
            print('Shuffling lines')
            shuffle(idx)

            for i, row_idx in enumerate(idx):
                item_path = df[0][row_idx]

                item_classes = []
                for j in range(1, df.shape[1]):
                    item_classes.append(df[j][row_idx])

                process_item(data_dir, data_shape, item_classes, item_path, x, y, i, df)
        else:
            for i, row in df.iterrows():
                item_path = row[0]

                item_classes = []
                for j in range(1, df.shape[1]):
                    item_classes.append(row[j])

                process_item(data_dir, data_shape, item_classes, item_path, x, y, i, df)

        x = np.array(x).astype(np.float32) / 255.
        x -= 0.5
        x_train_val.append(x)
        y_train_val.append(np.array(y))

    print('train_data:', x_train_val[0].shape, y_train_val[0].shape)
    print('val_data:', x_train_val[1].shape, y_train_val[1].shape)

    return (x_train_val[0], y_train_val[0]), (x_train_val[1], y_train_val[1])


def __load_specific_data(data_dir, data_shape, filename):
    print('Loading data from ' + data_dir)

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

    print('val_data:', x.shape, y.shape)

    return x, y


def load_train_data(data_dir, data_shape):
    return __load_specific_data(data_dir, data_shape, 'train.txt')


def load_val_data(data_dir, data_shape):
    return __load_specific_data(data_dir, data_shape, 'val.txt')


def __get_random_case(data_dir, filename):
    file_path = os.path.join(data_dir, filename)

    df = pd.read_csv(file_path, sep=' ', header=None)
    r = randint(0, df.shape[0] - 1)

    img_path = os.path.join(data_dir, df[0][r])
    img_class = int(df[1][r])

    return img_path, img_class


def get_random_train_case(data_dir):
    return __get_random_case(data_dir, 'train.txt')


def get_random_val_case(data_dir):
    return __get_random_case(data_dir, 'val.txt')


def load_prepared_image(path, data_shape):
    img = img_as_float(io.imread(path))
    img = transform.resize(img, data_shape)
    img -= 0.5
    img = np.expand_dims(img, axis=-1)
    return img


def __get_batches(data_dir, batch_size, filename):
    print('Getting batches of ' + filename)
    file_path = os.path.join(data_dir, filename)

    batches = []
    batch = []
    df = pd.read_csv(file_path, sep=' ', header=None)

    counter = 0
    for _, row in df.iterrows():
        item_path = row[0]
        item_classes = np.array(row[1:]).astype(int)

        batch.append((item_path, item_classes))

        counter += 1
        if counter == batch_size:
            batches.append(batch.copy())
            batch = []
            counter = 0

    if counter > 0:
        batches.append(batch)

    return batches, df.shape[0]


def get_train_batches(data_dir, batch_size, num_classes):
    return __get_batches(data_dir, batch_size, 'train_%icl.txt' % num_classes)


def get_val_batches(data_dir, batch_size, num_classes):
    return __get_batches(data_dir, batch_size, 'val_%icl.txt' % num_classes)


def load_batch(data_dir, data_shape, batch):
    x = []
    y = []

    for item_path, item_class in batch:
        process_item(data_dir, data_shape, item_class, item_path, x, y, 1, np.zeros((2, )))

    x = np.array(x).astype(np.float32) / 255.
    x -= 0.5
    y = np.array(y)

    return x, y

