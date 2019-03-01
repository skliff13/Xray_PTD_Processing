import os
import pickle
import numpy as np
from keras.models import load_model, Model
from keras_applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras_applications.resnet50 import ResNet50

from prediction_settings import XrayPredictionSettings


class ModelsLoader:
    corrs = None
    map_layer_model = None
    desc_layer_model = None
    cls_model = None
    segm_model = None
    classifiers = None

    @staticmethod
    def parse_model_type(model_type_name):
        if model_type_name == 'VGG16':
            return VGG16
        elif model_type_name == 'VGG19':
            return VGG19
        elif model_type_name == 'InceptionV3':
            return InceptionV3
        elif model_type_name == 'InceptionResNetV2':
            return InceptionResNetV2
        elif model_type_name == 'ResNet50':
            return ResNet50
        else:
            print('Unknown net model: ' + model_type_name)
            exit(1)
            return None

    def load_models(self, prediction_settings):
        s: XrayPredictionSettings = prediction_settings

        model_type = self.parse_model_type(s.model_type)
        cls_model = model_type(weights=None, include_top=True, input_shape=(s.image_sz, s.image_sz, 1), classes=2)
        map_layer_model = Model(inputs=cls_model.input, outputs=cls_model.get_layer(s.map_layer_name).output)
        desc_layer_model = Model(inputs=cls_model.input, outputs=cls_model.get_layer(s.desc_layer_name).output)

        corrs_path = s.weights_path[:-5] + '_' + s.map_layer_name + '_corrs.txt'
        print('Loading corrs ' + corrs_path)
        corrs = np.loadtxt(corrs_path)
        corrs[np.isnan(corrs)] = 0
        corrs = np.sign(corrs) * np.square(corrs)

        print('Loading weights ' + s.weights_path)
        cls_model.load_weights(s.weights_path)

        print('Loading model from ' + s.segm_model_path)
        segm_model = load_model(s.segm_model_path)

        classifiers = {}
        print('Loading trained classifiers from ' + s.classifiers_dir)
        for class_name in s.class_names:
            path = os.path.join(s.classifiers_dir, 'logit-%s.pickle' % class_name)
            classifier = pickle.load(open(path, 'rb'))
            classifiers[class_name] = classifier

        self.cls_model = cls_model
        self.map_layer_model = map_layer_model
        self.desc_layer_model = desc_layer_model
        self.corrs = corrs
        self.segm_model = segm_model
        self.classifiers = classifiers

        return self
