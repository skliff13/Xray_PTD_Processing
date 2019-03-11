import os
import pandas as pd


def main():
    data_dir = '/home/skliff13/work/PTD_Xray/datasets/abnormal_lungs/v2.0'

    # class_names = ['class_number', 'pneumonia', 'tuberculosis']
    class_names = ['class_number', 'bronchitis', 'emphysema', 'fibrosis', 'focal_shadows', 'pneumonia',
                   'pneumosclerosis', 'tuberculosis']

    num_classes = len(class_names)

    classes_of_case = read_classes_to_dict(class_names)

    for train_val in ['train', 'val']:
        print('Processing ' + train_val)

        df = pd.read_csv(os.path.join(data_dir, train_val + '.txt'), header=None, sep=' ')

        paths = []
        class_numbers = []
        for _ in range(len(class_names)):
            class_numbers.append([])

        for i, row in df.iterrows():
            if i % 1000 == 0:
                print('%i / %i' % (i, df.shape[0]))

            path = row[0]
            fn = os.path.split(path)[-1][7:-11]
            l = classes_of_case[fn]

            paths.append(path)
            for j in range(len(class_names)):
                class_numbers[j].append(l[j])

        out_path = os.path.join(data_dir, (train_val + '_%icl.txt') % num_classes)

        d = {'a_path': paths}
        for j in range(len(class_names)):
            column_name = 'b_class%03i' % j
            d[column_name] = class_numbers[j]
            print('Cases with label', j, ':', sum(class_numbers[j]))

        print('Saving to ' + out_path)
        pd.DataFrame.from_dict(d).to_csv(out_path, sep=' ', index=None, header=None)


def read_classes_to_dict(class_names):
    print('Reading study group classes to dictionary')
    df = pd.read_csv('../data/study_group_class_abnormal_lungs.txt')
    d = {}
    for i, row in df.iterrows():
        if i % 10000 == 0:
            print('%i / %i' % (i, df.shape[0]))

        l = []
        for class_name in class_names:
            l.append(row[class_name])

        d[row['filename']] = l
    return d


if __name__ == '__main__':
    main()
