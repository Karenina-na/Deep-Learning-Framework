from .tp_tn_fp_fn import TP_TN_FP_FN


# 利用混淆矩阵阴性预测值（npv）
def npv(confusion_matrix):
    """
    - confusion_matrix : 混淆矩阵
    """
    TP, TN, FP, FN = TP_TN_FP_FN(confusion_matrix)
    return TN / (TN + FN+1e-10)
