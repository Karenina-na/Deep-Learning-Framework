# 利用混淆矩阵敏感度
def sensitivity(confusion_matrix):
    """
    - confusion_matrix : 混淆矩阵
    """
    TP, TN, FP, FN = TP_TN_FP_FN(confusion_matrix)
    return TP / (TP + FN)