from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize


# 绘制PR曲线
def plot_PR_curve(test, output_prob, classes):
    """
    画PR曲线
    :param test:    测试集
    :param output_prob:     模型输出的概率
    :param classes:    类别
    :return:
    """
    # Binarize the output
    y_test = label_binarize(test, classes=classes)
    n_classes = y_test.shape[1]
    y_score = output_prob
    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")
    # PR AUC
    plt.figure(dpi=300)
    lw = 2
    plt.plot(recall["micro"], precision["micro"],
             label='micro-average PR curve (area = {0:0.2f})'
                   ''.format(average_precision["micro"]))
    for i in range(n_classes):
        plt.plot(recall[i], precision[i],
                 label='PR curve of class {0} (area = {1:0.2f})'
                       ''.format(i, average_precision[i]))
    bep = brentq(lambda x: x - interp1d(recall["micro"], precision["micro"])(x), 0., 1.)
    plt.plot([bep], [interp1d(recall["micro"], precision["micro"])(bep)], marker='o', markersize=5, color="red")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # ================================
    # plt.title('Some extension of Precision-Recall curve to multi-class \n (BEP = %0.4f)' % bep)
    plt.title(('BEP=%0.4f' % bep), fontsize=18, fontweight='bold')  # 标题
    plt.legend(loc="lower left", prop={'weight': 'bold', "size": 9})  # 图例
    # -------------------------------
    plt.xticks(fontweight='bold', fontsize=12)  # 横轴坐标刻度
    plt.xlim([0.0, 1.0])  # 横轴坐标范围
    plt.xlabel('Recall', fontsize=20, fontweight='bold')  # 横轴坐标名称
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)  # 纵轴坐标名称位置

    plt.yticks(fontweight='bold', fontsize=12)  # 纵轴坐标刻度
    plt.ylim([0.0, 1.05])  # 纵轴坐标范围
    plt.ylabel('Precision', fontsize=20, fontweight='bold')  # 纵轴坐标名称
    plt.gca().xaxis.set_label_coords(0.5, -0.1)  # 横轴坐标名称位置
    # ================================
    plt.tight_layout()
    plt.show()


def plot_PR_curve_Signal(test, output_prob):
    """
    画PR曲线
    :param test:   测试集
    :param output_prob:    模型输出的概率
    :return:
    """
    plt.figure(dpi=300)
    precision, recall, thresholds = precision_recall_curve(test, output_prob[:, 1])
    average_precision = average_precision_score(test, output_prob[:, 1])
    plt.figure(dpi=300)
    lw = 2
    plt.plot(recall, precision, color='darkorange', lw=lw,
             label='PR curve (area = %0.4f)' % average_precision)
    bep = brentq(lambda x: x - interp1d(recall, precision)(x), 0., 1.)
    plt.plot([bep], [interp1d(recall, precision)(bep)], marker='o', markersize=5, color="red")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # ================================
    # plt.title('Precision-Recall curve (BEP = %0.4f)' % bep, fontsize=18, fontweight='bold')
    plt.title(('BEP=%0.4f' % bep), fontsize=18, fontweight='bold')  # 标题
    plt.legend(loc="lower left", prop={'weight': 'bold', "size": 14})  # 图例
    # -------------------------------
    plt.xticks(fontweight='bold', fontsize=14)  # 横轴坐标刻度
    plt.xlim([0.0, 1.0])  # 横轴坐标范围
    plt.xlabel('Recall', fontsize=20, fontweight='bold')  # 横轴坐标名称
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)  # 纵轴坐标名称位置

    plt.yticks(fontweight='bold', fontsize=14)  # 纵轴坐标刻度
    plt.ylim([0.0, 1.05])  # 纵轴坐标范围
    plt.ylabel('Precision', fontsize=20, fontweight='bold')  # 纵轴坐标名称
    plt.gca().xaxis.set_label_coords(0.5, -0.1)  # 横轴坐标名称位置
    # ================================
    plt.tight_layout()
    plt.show()
