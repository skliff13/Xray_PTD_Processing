import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.visible_device_list = "1"
set_session(tf.Session(config=config))
import pydicom
import numpy as np
from skimage import io, color, exposure, morphology

from imutils import imresize, normalize_by_lung_mean_std
from prediction_settings import XrayPredictionSettings
from models_loader import ModelsLoader
from cropping import Cropping


class XrayPredictor:
    def __init__(self, setup_file_path, multi_label=False):
        self.prediction_setting = XrayPredictionSettings().load_setup(setup_file_path)
        self.models = ModelsLoader().load_models(self.prediction_setting, multi_label=multi_label)
        self.img_original = None
        self.mask = None
        self.img_roi = None

    def load_and_predict_image(self, input_image_path):
        self.img_original = self.load_original_image(input_image_path)

        img_gray = self.convert_to_gray(self.img_original)

        preview = self.make_preview(img_gray)

        lungs = self.segment_lungs(preview)

        img_normalized, mask, img_roi, mask_roi, cropping = self.normalize_and_crop(img_gray, lungs)

        desc, heat_map, prob = self.infer_neural_net(img_roi)

        predictions = self.predict_classes(desc, prob)

        rgb = self.make_colored(img_normalized, mask, heat_map, cropping)

        self.mask = mask
        self.img_roi = img_roi
        return predictions, rgb, img_normalized

    @staticmethod
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

    @staticmethod
    def convert_to_gray(img_original):
        if len(img_original.shape) > 2:
            if img_original.shape[2] == 1:
                img_gray = img_original[:, :, 0].copy()
            elif img_original.shape[2] == 3:
                img_gray = color.rgb2gray(img_original)
            elif img_original.shape[2] == 4:
                img_gray = color.rgb2gray(img_original[:, :, 0:3])
            else:
                raise Exception('Unsupported number of channels of the input image: ' + img_original.shape[2])
        else:
            img_gray = img_original.copy()

        img_gray = img_gray.astype(np.float32)

        return img_gray

    @staticmethod
    def make_preview(img_gray):
        preview_size = 256

        preview = imresize(img_gray, (preview_size, preview_size))
        preview = exposure.equalize_hist(preview)

        preview[preview < 0] = 0
        preview[preview > 1] = 1

        return preview

    def segment_lungs(self, preview):
        def remove_small_regions(img, size):
            img = morphology.remove_small_objects(img, size)
            img = morphology.remove_small_holes(img, size)
            return img

        x = preview.copy()

        x -= x.mean()
        x /= x.std()
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=-1)

        segm_model = self.models.segm_model
        lungs = segm_model.predict(x, batch_size=1)[..., 0].reshape(preview.shape)
        lungs = lungs > 0.5
        lungs = remove_small_regions(lungs, 0.02 * np.prod(preview.shape))

        return lungs

    @staticmethod
    def get_cropping(mask_bw):
        proj_x = np.sum(mask_bw, axis=0).flatten()
        proj_y = np.sum(mask_bw, axis=1).flatten()

        d = min(mask_bw.shape) // 20
        x_low = max(0, np.where(proj_x > 0)[0][0] - d)
        x_high = min(mask_bw.shape[1], np.where(proj_x > 0)[0][-1] + d)
        y_low = max(0, np.where(proj_y > 0)[0][0] - d)
        y_high = min(mask_bw.shape[0], np.where(proj_y > 0)[0][-1] + d)

        return Cropping(x_low, x_high, y_low, y_high)

    def normalize_and_crop(self, img_gray, lungs):
        image_sz = self.prediction_setting.image_sz

        mask = imresize(lungs, img_gray.shape, order=0)
        img_normalized = normalize_by_lung_mean_std(img_gray, mask)
        img_normalized[img_normalized < 0] = 0
        img_normalized[img_normalized > 1] = 1

        cropping = self.get_cropping(mask)
        img_roi = cropping.crop_image(img_normalized)
        mask_roi = cropping.crop_image(mask)

        img_roi = imresize(img_roi, (image_sz, image_sz))
        mask_roi = imresize(mask_roi, (image_sz, image_sz), order=0)

        return img_normalized, mask, img_roi, mask_roi, cropping

    def infer_neural_net(self, img_roi):
        m: ModelsLoader = self.models
        s: XrayPredictionSettings = self.prediction_setting

        image_sz = img_roi.shape[0]

        x = img_roi - 0.5
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=-1)

        print('Evaluating net')
        prob = m.cls_model.predict(x, batch_size=1)[0, 1]
        map_layer_output = m.map_layer_model.predict(x, batch_size=1)

        if s.desc_layer_name == s.map_layer_name:
            desc = np.mean(map_layer_output, axis=(1, 2))
        else:
            desc = m.desc_layer_model.predict(x, batch_size=1)

        heat_map = np.matmul(map_layer_output, m.corrs)[0, ...]
        heat_map = imresize(heat_map, (image_sz, image_sz)) * s.multiplier
        heat_map[heat_map < 0] = 0
        heat_map[heat_map > 1] = 1

        return desc, heat_map, prob

    def predict_classes(self, desc, prob):
        s: XrayPredictionSettings = self.prediction_setting
        m: ModelsLoader = self.models

        predictions = {'abnormal_lungs': round(float(prob), 3)}
        for class_name in s.class_names:
            classifier = m.classifiers[class_name]
            prediction = classifier.predict_proba(desc)[:, 1]
            predictions[class_name] = round(float(prediction[0]), 3)

        return predictions

    @staticmethod
    def make_colored(img_normalized, mask, heat_map, cropping):
        sz = img_normalized.shape
        hsv = np.zeros((sz[0], sz[1], 3))

        v = img_normalized
        v[v < 0] = 0
        v[v > 1] = 1
        hsv[:, :, 2] = v
        hsv[:, :, 1] = mask * 0.5

        x_low, x_high, y_low, y_high = cropping.unpack_values()

        map = img_normalized * 0
        map[y_low:y_high, x_low:x_high] = imresize(heat_map, (y_high - y_low, x_high - x_low))
        map[map < 0] = 0
        map[map > 1] = 1
        hsv[:, :, 0] = 0.7 * (1 - map)

        rect_hue = 0.8
        rect_sat = 1
        d = 3
        hsv[y_low:y_low + d, x_low:x_high, 0] = rect_hue
        hsv[y_high:y_high + d, x_low:x_high, 0] = rect_hue
        hsv[y_low:y_high, x_low:x_low + d, 0] = rect_hue
        hsv[y_low:y_high, x_high:x_high + d, 0] = rect_hue

        hsv[y_low:y_low + d, x_low:x_high, 1] = rect_sat
        hsv[y_high:y_high + d, x_low:x_high, 1] = rect_sat
        hsv[y_low:y_high, x_low:x_low + d, 1] = rect_sat
        hsv[y_low:y_high, x_high:x_high + d, 1] = rect_sat

        rgb = color.hsv2rgb(hsv)

        return rgb
