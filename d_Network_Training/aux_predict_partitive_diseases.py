import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from scipy.stats.stats import pearsonr
import numpy as np


def read_list_as_dict(list_path, class_names):
    df = pd.read_csv(list_path)

    d = {}
    for _, row in df.iterrows():
        key = row['filename']
        values = []
        for class_name in class_names:
            values.append(row[class_name])
        values = np.array(values)

        d[key] = values

    return d


def compose_y(paths, d, class_names):
    y = np.zeros((len(paths), len(class_names)))

    for i, path in enumerate(paths):
        filename = path[11:-11]
        y[i, :] = d[filename]

    return y

def main():
    data_dir = '/home/skliff13/work/PTD_Xray/datasets/abnormal_lungs/v2.0'
    model_path = 'models/abnormal_lungs_v2.0_Sz299_InceptionV3_RMSprop_Ep50_Lr1.0e-04_Auc0.880.hdf5'
    layer_name = 'mixed10'
    list_path = '../data/study_group_class_abnormal_lungs.txt'

    model_filename = os.path.split(model_path)[-1]
    predictions_dir = os.path.join(data_dir, 'predictions', model_filename[:-5])

    file_path = os.path.join(predictions_dir, layer_name + '_averages_train.txt')
    print('Loading ' + file_path)
    df_train = pd.read_csv(file_path)
    file_path = os.path.join(predictions_dir, layer_name + '_averages_val.txt')
    print('Loading ' + file_path)
    df_val = pd.read_csv(file_path)

    paths_train = df_train['path']
    paths_val = df_val['path']

    cols = df_train.columns.tolist()
    x_train = df_train[cols[1:]].values
    x_val = df_val[cols[1:]].values

    class_names = ['healthy', 'bronchitis', 'emphysema', 'fibrosis','focal_shadows', 'pneumonia', 'pneumosclerosis',
                   'tuberculosis', 'is_male']
    d = read_list_as_dict(list_path, class_names)

    ys_train = compose_y(paths_train, d, class_names)
    ys_val = compose_y(paths_val, d, class_names)

    each_training = 10
    print('Reducing training data by', each_training)
    for class_idx, class_name in enumerate(class_names):
        y_train = ys_train[:, class_idx]
        y_val = ys_val[:, class_idx]

        reg = LinearRegression().fit(x_train[::each_training, :], y_train[::each_training])
        pred_train = reg.predict(x_train)
        pred_val = reg.predict(x_val)

        auc_train = roc_auc_score(y_train, pred_train)
        auc_val = roc_auc_score(y_val, pred_val)

        print('Class: %s, AUC %.3f (%.3f on training)' % (class_name, auc_val, auc_train))


main()