import os
import keras
from glob import glob
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from inception_v1 import InceptionV1


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
            pattern = 'model_Sz%i_%s_%s_Ep%i_Lr%.1e*.hdf5'
            pattern = pattern % (image_sz, model_type.__name__, optimizer.__name__, epochs, learning_rate)

            files = glob(pattern)
            if not files:
                args = ['python3', 'train_model.py', str(batch_size), data_dir, str(epochs), str(image_sz),
                        str(learning_rate), model_type.__name__, str(num_classes), optimizer.__name__]
                cmd = ' '.join(args)
                print(cmd)
                os.system(cmd)


if __name__ == '__main__':
    main()
