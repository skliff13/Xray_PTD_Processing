import os
from glob import glob


def main():
    num_classes = 2
    # image_sz = 224
    # model_type = 'InceptionV3'
    data_dir = '/home/skliff13/work/PTD_Xray/datasets/tuberculosis/v2.2'
    epochs = 3
    batch_size = 32
    learning_rate = 1e-4
    optimizer = 'RMSprop'
    crop_to = 224

    # for optimizer in ['SGD', 'Adam', 'RMSprop']:
    # for model_type in ['VGG16', 'VGG19', 'InceptionV3', 'InceptionV1', 'InceptionResNetV2', 'ResNet50']:
    for model_type in ['InceptionV1']:
        for image_sz in [256]:
            pattern = 'model_Sz%i_%s_%s_Ep%i_Lr%.1e*.hdf5'
            pattern = pattern % (image_sz, model_type, optimizer, epochs, learning_rate)

            files = glob(pattern)
            if not files:
                args = ['python3', 'train_model.py', str(batch_size), data_dir, str(epochs), str(image_sz),
                        str(learning_rate), model_type, str(num_classes), optimizer, str(crop_to)]
                cmd = ' '.join(args)
                print(cmd)
                os.system(cmd)


if __name__ == '__main__':
    main()
