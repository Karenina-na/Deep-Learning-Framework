from .tp_tn_fp_fn import TP_TN_FP_FN


# 利用混淆矩阵假阳性率（FPR）
def fpr(confusion_matrix):
    """
    - confusion_matrix : 混淆矩阵
    """
    TP, TN, FP, FN = TP_TN_FP_FN(confusion_matrix)
    return FP / (FP + TN+1e-10)
