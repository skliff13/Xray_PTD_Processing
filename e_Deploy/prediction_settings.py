import json


class XrayPredictionSettings:
    image_sz = None
    weights_path = None
    layer_name = None
    model_type = None
    classifiers_dir = None
    class_names = None
    multiplier = None
    segm_model_path = None

    def load_setup(self, json_file_path):
        with open(json_file_path, 'r') as f:
            print('Loading setup from ' + json_file_path)
            d = json.load(f)

            for key in d:
                print('#   ', key, ':', d[key])

            self.image_sz = d['image_sz']
            self.weights_path = d['weights_path']
            self.map_layer_name = d['map_layer_name']
            self.desc_layer_name = d['desc_layer_name']
            self.model_type = d['model_type']
            self.classifiers_dir = d['classifiers_dir']
            self.class_names = d['class_names']
            self.multiplier = d['multiplier']
            self.segm_model_path = d['segm_model_path']

        return self
