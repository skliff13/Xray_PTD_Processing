import os
import json
from glob import glob


def main():
    num_classes = 2
    image_sz = 224
    # model_type = 'InceptionV3'
    data_dir = '/home/skliff13/work/PTD_Xray/datasets/tuberculosis/v2.3'
    # data_dir = '/home/skliff13/work/PTD_Xray/datasets/abnormal_lungs/v2.0'
    epochs = 2
    batch_size = 32
    learning_rate = 1e-4
    # optimizer = 'RMSprop'
    crop_to = -1

    json_path = os.path.join(data_dir, 'config.json')
    print('Reading config from ' + json_path)
    with open(json_path, 'r') as f:
        config = json.load(f)

    for optimizer in ['RMSprop']:
    # for model_type in ['VGG16', 'VGG19', 'InceptionV3', 'InceptionV1', 'InceptionResNetV2', 'ResNet50']:
    # for image_sz in [256]:
        for model_type in ['VGG16']:
            pattern = 'models/%s_Sz%i_%s_%s_Ep%i_Lr%.1e*.hdf5'
            pattern = pattern % (config['dataset'], image_sz, model_type, optimizer, epochs, learning_rate)

            files = glob(pattern)
            if not files:
                args = ['CUDA_VISIBLE_DEVICES=1', 'python3', 'train_model.py', str(batch_size), data_dir, str(epochs),
                        str(image_sz), str(learning_rate), model_type, str(num_classes), optimizer, str(crop_to)]
                cmd = ' '.join(args)
                print(cmd)
                os.system(cmd)


if __name__ == '__main__':
    main()
