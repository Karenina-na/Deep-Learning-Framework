from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize


# 绘制PR曲线
def plot_PR(test, output_prob, classes):
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
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Some extension of Precision-Recall curve to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def plot_PR_Signal(test, output_prob):
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
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.show()