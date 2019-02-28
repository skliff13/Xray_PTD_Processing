import os
# if __name__ == '__main__':
#     os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.visible_device_list = "1"
set_session(tf.Session(config=config))
import pickle
import pydicom
import numpy as np
from skimage import io, color, exposure, morphology
from keras.models import load_model, Model
from keras.applications.inception_v3 import InceptionV3

from imutils import imresize, normalize_by_lung_mean_std


def load_original_image(input_image_path):
    print('Loading image from ' + input_image_path)

    ext = os.path.splitext(input_image_path)[-1].lower()
    if ext in ['.jpg', '.png', '.bmp', '.jpeg']:
        img_original = io.imread(input_image_path)
    elif ext in ['.dcm']:
        dcm = pydicom.dcmread(input_image_path)
        img_original = dcm.pixel_array
    else:
        raise Exception('Unsupported input image extension: ' + ext)

    print('Loaded image (%i x %i)' % (img_original.shape[0], img_original.shape[1]))
    return img_original


def convert_to_gray(img_original):
    if len(img_original.shape) > 2:
        if img_original.shape[2] == 1:
            img_gray = img_original[:, :, 0].copy()
        elif img_original.shape[2] == 3:
            img_gray = color.rgb2gray(img_original)
        else:
            raise Exception('Unsupported number of channels of the input image: ' + img_original.shape[2])
    else:
        img_gray = img_original.copy()

    img_gray = img_gray.astype(np.float32)

    return img_gray


def make_preview(img_gray):
    preview_size = 256

    preview = imresize(img_gray, (preview_size, preview_size))
    preview = exposure.equalize_hist(preview)

    preview[preview < 0] = 0
    preview[preview > 1] = 1

    return preview


def remove_small_regions(img, size):
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img


def segment_lungs(preview):
    x = preview.copy()

    x -= x.mean()
    x /= x.std()
    x = np.expand_dims(x, axis=0)
    x = np.expand_dims(x, axis=-1)

    model_path = '../c_Process_Files/lungs_segmentation_tf/trained_model.hdf5'
    print('Loading model from ' + model_path)
    UNet = load_model(model_path)

    lungs = UNet.predict(x, batch_size=1)[..., 0].reshape(preview.shape)
    lungs = lungs > 0.5
    lungs = remove_small_regions(lungs, 0.02 * np.prod(preview.shape))

    return lungs


def get_cropping(mask_bw):
    proj_x = np.sum(mask_bw, axis=0).flatten()
    proj_y = np.sum(mask_bw, axis=1).flatten()

    d = min(mask_bw.shape) // 20
    x_low = max(0, np.where(proj_x > 0)[0][0] - d)
    x_high = min(mask_bw.shape[1], np.where(proj_x > 0)[0][-1] + d)
    y_low = max(0, np.where(proj_y > 0)[0][0] - d)
    y_high = min(mask_bw.shape[0], np.where(proj_y > 0)[0][-1] + d)

    return x_low, x_high, y_low, y_high


def normalize_and_crop(img_gray, lungs):
    mask = imresize(lungs, img_gray.shape, order=0)
    img_normalized = normalize_by_lung_mean_std(img_gray, mask)
    img_normalized[img_normalized < 0] = 0
    img_normalized[img_normalized > 1] = 1

    x_low, x_high, y_low, y_high = get_cropping(mask)
    img_roi = img_normalized[y_low:y_high, x_low:x_high]
    mask_roi = mask[y_low:y_high, x_low:x_high]

    out_size = 299
    img_roi = imresize(img_roi, (out_size, out_size))
    mask_roi = imresize(mask_roi, (out_size, out_size), order=0)

    return img_normalized, mask, img_roi, mask_roi, x_low, x_high, y_low, y_high


def infer_neural_net(img_roi):
    image_sz = img_roi.shape[0]
    model = InceptionV3(weights=None, include_top=True, input_shape=(image_sz, image_sz, 1), classes=2)

    weights_path = '../d_Network_Training/models/' \
                   'abnormal_lungs_v2.0_Sz299_InceptionV3_RMSprop_Ep50_Lr1.0e-04_Auc0.880.hdf5'
    print('Loading model ' + weights_path)
    model.load_weights(weights_path)

    layer_name = 'mixed10'

    corrs_path = weights_path[:-5] + '_' + layer_name + '_corrs.txt'
    print('Loading corrs ' + corrs_path)
    corrs = np.loadtxt(corrs_path)
    corrs = np.sign(corrs) * np.square(corrs)

    x = img_roi - 0.5
    x = np.expand_dims(x, axis=0)
    x = np.expand_dims(x, axis=-1)

    print('Evaluating activations')
    layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    layer_output = layer_model.predict(x, batch_size=1)

    layer_averages = np.mean(layer_output, axis=(1, 2))

    heat_map = np.matmul(layer_output, corrs)[0, ...]
    heat_map = imresize(heat_map, (image_sz, image_sz)) / 1000
    heat_map[heat_map < 0] = 0
    heat_map[heat_map > 1] = 1

    return layer_averages, heat_map


def predict_classes(layer_averages):
    class_names = ['healthy', 'bronchitis', 'emphysema', 'fibrosis', 'focal_shadows', 'pneumonia', 'pneumosclerosis',
                   'tuberculosis']
    classifiers_dir = '../d_Network_Training/classifiers/' \
                      'abnormal_lungs_v2.0_Sz299_InceptionV3_RMSprop_Ep50_Lr1.0e-04_Auc0.880'

    print('Loading trained classifiers from ' + classifiers_dir)
    predictions = {}
    for class_name in class_names:
        path = os.path.join(classifiers_dir, 'logit-%s.pickle' % class_name)
        classifier = pickle.load(open(path, 'rb'))

        prediction = classifier.predict_proba(layer_averages)[:, 1]
        predictions[class_name] = prediction[0]

    return predictions


def make_colored(img_normalized, mask, heat_map, x_low, x_high, y_low, y_high):
    sz = img_normalized.shape
    hsv = np.zeros((sz[0], sz[1], 3))

    v = img_normalized
    v[v < 0] = 0
    v[v > 1] = 1
    hsv[:, :, 2] = v
    hsv[:, :, 1] = mask * 0.5

    map = img_normalized * 0
    map[y_low:y_high, x_low:x_high] = imresize(heat_map, (y_high - y_low, x_high - x_low))
    map[map < 0] = 0
    map[map > 1] = 1
    hsv[:, :, 0] = 0.7 * (1 - map)

    rect_hue = 0.8
    rect_sat = 1
    d = 3
    hsv[y_low:y_low + d, x_low:x_high, 0] = rect_hue
    hsv[y_low:y_low + d, x_low:x_high, 0] = rect_hue
    hsv[y_low:y_high, x_low:x_low + d, 0] = rect_hue
    hsv[y_low:y_high, x_high:x_high + d, 0] = rect_hue

    hsv[y_low:y_low + d, x_low:x_high, 1] = rect_sat
    hsv[y_low:y_low + d, x_low:x_high, 1] = rect_sat
    hsv[y_low:y_high, x_low:x_low + d, 1] = rect_sat
    hsv[y_low:y_high, x_high:x_high + d, 1] = rect_sat

    rgb = color.hsv2rgb(hsv)

    return rgb


def main():
    input_image_path = 'test_data/tb_03.png'
    # input_image_path = 'test_data/pneumothorax_01.jpg'
    # input_image_path = 'test_data/pneum_01.png'
    # input_image_path = 'test_data/nipple_shadows_01.jpg'
    # input_image_path = 'test_data/crdf_id011_01.dcm'
    # input_image_path = 'test_data/crdf_id100_03.dcm'

    img_original = load_original_image(input_image_path)
    img_gray = convert_to_gray(img_original)
    preview = make_preview(img_gray)
    lungs = segment_lungs(preview)
    img_normalized, mask, img_roi, mask_roi, x_low, x_high, y_low, y_high = normalize_and_crop(img_gray, lungs)
    layer_averages, heat_map = infer_neural_net(img_roi)
    predictions = predict_classes(layer_averages)
    rgb = make_colored(img_normalized, mask, heat_map, x_low, x_high, y_low, y_high)

    print('Predictions')
    for class_name in predictions:
        print('%s: %.02f' % (class_name, predictions[class_name]))

    io.imshow_collection((img_normalized, rgb))
    io.show()


if __name__ == '__main__':
    main()
