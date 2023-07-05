import itertools

import matplotlib.pyplot as plt
import numpy as np


# 画混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    - cm :  the value of the confusion matrix calculated
    - classes : the column corresponding to each row and column in the confusion matrix
    - normalize :   True: show percentage, False: show number
    """
    plt.figure(dpi=300)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    # matplotlib problem, if not add the following code, the confusion matrix can only be displayed half, some versions of matplotlib do not need the following code, try it separately
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment='center',
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    classes = [0, 1, 2, 3, 4, 5]
    cm = np.array([[10, 0, 0, 0, 0, 0],
                   [0, 10, 0, 0, 0, 0],
                   [0, 0, 10, 0, 0, 0],
                   [0, 0, 0, 10, 0, 0],
                   [0, 0, 0, 0, 10, 0],
                   [0, 0, 0, 0, 0, 10]])
    plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)
