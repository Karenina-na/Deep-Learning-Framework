import itertools

import matplotlib.pyplot as plt
import numpy as np


# 画混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Greens):
    """
    - cm :  the value of the confusion matrix calculated
    - classes : the column corresponding to each row and column in the confusion matrix
    - normalize :   True: show percentage, False: show number
    """
    plt.figure(dpi=500)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # matplotlib problem, if not add the following code, the confusion matrix can only be displayed half, some versions of matplotlib do not need the following code, try it separately
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # ===============================
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment='center',
                 color="white" if cm[i, j] > thresh else "black", fontweight='bold', fontsize=20)
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置字体
    # ------------------------------- # 标题
    plt.title(title, fontsize=22, fontweight='bold')  # 标题
    # ------------------------------- # 色标
    cbar_ax = plt.colorbar()  # 显示色标
    cbar_ax.ax.tick_params(axis='y', labelsize=12, direction='in')  # 色标刻度字体大小粗细
    char_list = list(cbar_ax.ax.get_yticklabels())
    for i in range(len(char_list)):
        char_list[i].set_fontweight('bold')
    cbar_ax.set_label('Percentage', fontsize=16, fontweight='bold')  # 色标名称
    cbar_ax.ax.yaxis.set_label_coords(3, 0.5)  # 色标名称位置
    # ------------------------------- # 坐标轴
    tick_marks = np.arange(len(classes))

    plt.yticks(tick_marks, classes, fontweight='bold')  # 纵轴坐标名称
    plt.ylim(len(classes) - 0.5, -0.5)  # 纵轴坐标范围
    plt.ylabel('True label', fontsize=20, fontweight='bold')  # 纵轴坐标名称
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)  # 纵轴坐标名称位置

    plt.xticks(tick_marks, classes, fontweight='bold', rotation=45)  # 横轴坐标名称
    plt.xlim(-1, len(classes))  # 横轴坐标范围
    plt.xlabel('Predicted label', fontsize=20, fontweight='bold')  # 横轴坐标名称
    plt.gca().xaxis.set_label_coords(0.5, -0.1)  # 横轴坐标名称位置

    plt.tick_params(labelsize=12)  # 坐标轴刻度字体大小粗细
    # ===============================
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
