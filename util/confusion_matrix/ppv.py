from .TP_TN_FP_FN import TP_TN_FP_FN


# 利用混淆矩阵阳性预测值（PPV）
def ppv(confusion_matrix):
    """
    - confusion_matrix : 混淆矩阵
    """
    TP, TN, FP, FN = TP_TN_FP_FN(confusion_matrix)
    return TP / (TP + FP+1e-10)
