import numpy as np
from xray_predictor import XrayPredictor
from imutils import imresize
from prediction_settings import XrayPredictionSettings
from models_loader import ModelsLoader


class XrayPredictorMulti(XrayPredictor):
    def __init__(self, setup_file_path):
        XrayPredictor.__init__(self, setup_file_path, multi_label=True)

    def infer_neural_net(self, img_roi):
        m: ModelsLoader = self.models
        s: XrayPredictionSettings = self.prediction_setting

        image_sz = img_roi.shape[0]

        x = img_roi - 0.5
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=-1)

        print('Evaluating net')
        prob = m.cls_model.predict(x, batch_size=1)[0, :]
        map_layer_output = m.map_layer_model.predict(x, batch_size=1)

        heat_map = np.matmul(map_layer_output, m.corrs)[0, ...]
        heat_map = imresize(heat_map, (image_sz, image_sz)) * s.multiplier
        heat_map[heat_map < 0] = 0
        heat_map[heat_map > 1] = 1

        return None, heat_map, prob

    def predict_classes(self, _, prob):
        s: XrayPredictionSettings = self.prediction_setting

        predictions = {}
        counter = 0
        for class_name in s.class_names:
            prediction = prob[counter]
            predictions[class_name] = round(float(prediction), 3)
            counter += 1

        return predictions


