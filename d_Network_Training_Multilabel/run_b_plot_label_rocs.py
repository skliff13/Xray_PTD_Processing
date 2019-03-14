import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from mpl_toolkits import mplot3d


def plot3d_auc(y_val, pred, class_index):
    ax = plt.axes(projection='3d')
    p = pred[:, class_index]
    l = y_val[:, class_index]
    fpr, tpr, thre = roc_curve(l, p)
    ax.scatter3D(fpr, tpr, thre)
    plt.show()


def main():
    class_names = ['abnormal_lungs', 'pneumonia', 'tuberculosis']
    num_classes = len(class_names)
    data_dir = '/home/skliff13/work/uiip/datasets/abnormal_lungs/v2.0'

    weights_path = '../d_Network_Training_Multilabel/models/'
    # weights_path += 'abnormal_lungs_v2.0_8cl_Sz224_VGG16_Adam_Ep30_Lr1.0e-05_MeanAuc0.848.hdf5'
    weights_path += 'abnormal_lungs_v2.0_Sz224_VGG16_Adam_Ep30_Lr1.0e-05_MeanAuc0.883.hdf5'

    df = pd.read_csv(os.path.join(data_dir, 'val_%icl.txt' % num_classes), header=None, sep=' ')
    y_val = df[df.columns[1:]].get_values()

    model_filename = os.path.split(weights_path)[-1]
    pred_dir = os.path.join(data_dir, 'predictions', model_filename[:-5])

    pred_path = os.path.join(pred_dir, 'pred_val.txt')
    print('Loading cached predictions from ' + pred_path)
    pred = pd.read_csv(pred_path, header=None).get_values()

    # plot3d_auc(y_val, pred, 1)
    # exit(13)

    lw = 2
    plt.figure(figsize=(4.5, 4.5), dpi=600)
    # plt.figure(figsize=(6, 6))
    # plt.plot([0, 1], [0, 1], color='lime', lw=lw, linestyle='--', label='baseline')

    line_styles = [':', '-.', '-']
    counter = 0

    for i, class_name in enumerate(class_names):
        fpr, tpr, thre = roc_curve(y_val[:, i], pred[:, i])

        print('\nClass: ' + class_name)
        q = tpr - fpr
        print('  max Quality at threshold %.4f' % thre[q == q.max()][0])
        print('  0.95 TPR at threshold %.4f (FPR = %.2f)' % (thre[tpr > 0.95][0], fpr[tpr > 0.95][0]))
        print('  0.90 TPR at threshold %.4f (FPR = %.2f)' % (thre[tpr > 0.9][0], fpr[tpr > 0.9][0]))
        print('  0.80 TPR at threshold %.4f (FPR = %.2f)' % (thre[tpr > 0.8][0], fpr[tpr > 0.8][0]))
        print('  0.20 FPR at threshold %.4f (TPR = %.2f)' % (thre[fpr > 0.2][0], tpr[fpr > 0.2][0]))
        print('  0.10 FPR at threshold %.4f (TPR = %.2f)' % (thre[fpr > 0.1][0], tpr[fpr > 0.1][0]))
        print('  0.05 FPR at threshold %.4f (TPR = %.2f)' % (thre[fpr > 0.05][0], tpr[fpr > 0.05][0]))

        roc_auc = auc(fpr, tpr)

        ls = line_styles[counter]
        counter = (counter + 1) % len(line_styles)

        plt.plot(fpr, tpr, lw=lw, label=('%s (AUC %0.3f)' % (class_name, roc_auc)), linestyle=ls)
        # ax.plot3D(fpr, tpr, thre)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    # plt.title('ROC curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', prop={'size': 10})
    plt.savefig('fig.png')
    # plt.show()


if __name__ == '__main__':
    main()
