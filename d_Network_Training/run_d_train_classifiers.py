import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle
import json


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


def compose_data(paths, d, class_names):
    data = np.zeros((len(paths), len(class_names)))

    for i, path in enumerate(paths):
        filename = path[11:-11]
        data[i, :] = d[filename]

    return data


def main():
    with open('setup_vgg19_1.json', 'r') as f:
        d = json.load(f)

    data_dir = d['data_dir']
    weights_path = d['weights_path']
    desc_layer_name = d['desc_layer_name']
    list_path = d['list_path']
    class_names = d['class_names']

    weights_filename = os.path.split(weights_path)[-1]
    predictions_dir = os.path.join(data_dir, 'predictions', weights_filename[:-5])

    file_path = os.path.join(predictions_dir, desc_layer_name + '_descs_train.txt')
    print('Loading ' + file_path)
    df_train = pd.read_csv(file_path)
    file_path = os.path.join(predictions_dir, desc_layer_name + '_descs_val.txt')
    print('Loading ' + file_path)
    df_val = pd.read_csv(file_path)

    paths_train = df_train['path']
    paths_val = df_val['path']

    cols = df_train.columns.tolist()
    x_train = df_train[cols[1:]].values
    x_val = df_val[cols[1:]].values

    d = read_list_as_dict(list_path, class_names)

    ys_train = compose_data(paths_train, d, class_names)
    ys_val = compose_data(paths_val, d, class_names)

    # add_names = ['age', 'is_male']
    add_names = []
    if add_names:
        print('Using additional data:', add_names)
        d_add = read_list_as_dict(list_path, add_names)
        add_train = compose_data(paths_train, d_add, add_names)
        add_val = compose_data(paths_val, d_add, add_names)
        x_train = np.concatenate((x_train, add_train), axis=1)
        x_val = np.concatenate((x_val, add_val), axis=1)

    out_dir = os.path.join('classifiers', weights_filename[:-5])
    os.makedirs(out_dir, exist_ok=True)

    each_training = 1
    print('Reducing training data by', each_training)
    for class_idx, class_name in enumerate(class_names):
        y_train = ys_train[:, class_idx]
        y_val = ys_val[:, class_idx]

        classifier_model = LogisticRegression().fit(x_train[::each_training, :], y_train[::each_training])
        pred_train = classifier_model.predict_proba(x_train)[:, 1]
        pred_val = classifier_model.predict_proba(x_val)[:, 1]

        auc_train = roc_auc_score(y_train, pred_train)
        auc_val = roc_auc_score(y_val, pred_val)

        print('Class: %s, AUC %.3f (%.3f on training)' % (class_name, auc_val, auc_train))

        if add_names:
            add_str = '+'.join(add_names)
            store_path = os.path.join(out_dir, 'logit-%s+%s.pickle' % (class_name, add_str))
        else:
            store_path = os.path.join(out_dir, 'logit-%s.pickle' % class_name)
        pickle.dump(classifier_model, open(store_path, 'wb'))


main()
