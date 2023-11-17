from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


# 画ROC_AUC曲线
def ROC_AUC(test, output_prob, classes):
    """
    ROC AUC
    :param test:    测试集
    :param output_prob:     模型输出的概率
    :param classes:     类别
    :param show_error:  是否显示EER
    :return:
    """
    lw = 2
    # Binarize the output
    y_test = label_binarize(test, classes=classes)
    n_classes = y_test.shape[1]
    y_score = output_prob
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Plot of a ROC curve for a specific class
    plt.figure(dpi=300)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(classes[i], roc_auc[i]))
    eer = brentq(lambda x: 1. - x - interp1d(fpr["micro"], tpr["micro"])(x), 0., 1.)
    plt.plot([eer], [interp1d(fpr["micro"], tpr["micro"])(eer)], marker='o', markersize=5, color="red")
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # ================================
    # plt.title('Some extension of Receiver operating characteristic to multi-class \n (ERR=%0.4f)' % eer,
    #           fontsize=18, fontweight='bold')
    plt.legend(loc="lower right", prop={'weight': 'bold', "size": 9})
    # -------------------------------
    plt.xticks(fontweight='bold', fontsize=12)
    plt.xlim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=20, fontweight='bold')
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)  # 纵轴坐标名称位置

    plt.yticks(fontweight='bold', fontsize=12)
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate', fontsize=20, fontweight='bold')
    plt.gca().xaxis.set_label_coords(0.5, -0.1)  # 横轴坐标名称位置
    # ================================
    plt.tight_layout()
    plt.show()


def ROC_AUC_Signal(test, output_prob):
    """
    ROC AUC for 二分类
    :param test:   测试集
    :param output_prob:  模型输出的概率
    :return:
    """
    fpr, tpr, thresholds = roc_curve(test, output_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(dpi=300)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw,
             label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    plt.plot([eer], [interp1d(fpr, tpr)(eer)], marker='o', markersize=5, color="red")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # ================================
    # plt.title('Receiver operating characteristic (ERR=%0.4f)' % eer, fontsize=18, fontweight='bold')
    plt.title(('ERR=%0.4f' % eer), fontsize=18, fontweight='bold')
    plt.legend(loc="lower right", prop={'weight': 'bold', "size": 14})
    # -------------------------------
    plt.xticks(fontweight='bold', fontsize=14)
    plt.xlim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=20, fontweight='bold')
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)  # 纵轴坐标名称位置

    plt.yticks(fontweight='bold', fontsize=14)
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate', fontsize=20, fontweight='bold')
    plt.gca().xaxis.set_label_coords(0.5, -0.1)  # 横轴坐标名称位置
    # ================================
    plt.tight_layout()
    plt.show()
