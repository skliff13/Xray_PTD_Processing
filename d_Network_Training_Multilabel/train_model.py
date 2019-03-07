import os
import json
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.visible_device_list = "1"
set_session(tf.Session(config=config))
import keras
from sklearn.metrics import roc_auc_score
from keras_applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from vgg16_multilabel import VGG16
from keras.applications.vgg19 import VGG19
from keras_applications.resnet50 import ResNet50

from inception_v1 import InceptionV1
from data_gen import ModifiedDataGenerator
from load_data import load_data


def train_model(batch_size, data_dir, epochs, image_sz, learning_rate, model_type, num_classes, optimizer, crop_to):
    data_shape = (image_sz, image_sz)
    (x_train, y_train), (x_val, y_val) = load_data(data_dir, data_shape)

    if crop_to > 0:
        final_image_sz = crop_to
    else:
        final_image_sz = image_sz

    model = model_type(weights=None, include_top=True, input_shape=(final_image_sz, final_image_sz, 1),
                       classes=num_classes)

    json_path = os.path.join(data_dir, 'config.json')
    print('Reading config from ' + json_path)
    with open(json_path, 'r') as f:
        config = json.load(f)

    if not os.path.isdir('models'):
        os.mkdir('models')

    pattern = 'models/%s_Sz%i_%s_%s_Ep%i_Lr%.1e*.hdf5'
    pattern = pattern % (config['dataset'], image_sz, model_type.__name__, optimizer.__class__.__name__,
                         epochs, learning_rate)

    print('\n### Running training for ' + pattern + '\n')

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

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
    mean_auc = 0.0
    for j in range(num_classes):
        roc_auc = roc_auc_score(y_val[:, j].ravel(), predictions[:, j].ravel())
        print('Class %i AUC: %.03f' % (j, roc_auc))
        mean_auc += roc_auc / num_classes

    model_filename = pattern.replace('*', '_MeanAuc%.3f' % mean_auc)
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
    else:
        print('Unknown net model: ' + model_type)
        exit(1)

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
    os.sys.argv = 'train_model.py 16 /home/skliff13/work/PTD_Xray/datasets/abnormal_lungs/v2.0 30 224 0.00001 ' \
                  'VGG16 3 Adam -1'.split(' ')
    main()
