from .TP_TN_FP_FN import TP_TN_FP_FN


# 利用混淆矩阵F1值
def f1_score(confusion_matrix):
    """
    - confusion_matrix : 混淆矩阵
    """
    TP, TN, FP, FN = TP_TN_FP_FN(confusion_matrix)
    return 2 * TP / (2 * TP + FP + FN)
