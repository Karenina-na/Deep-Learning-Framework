# TP、TN、FP、FN
# TP：True Positive，真正例，预测为正例，实际为正例
# TN：True Negative，真负例，预测为负例，实际为负例
# FP：False Positive，假正例，预测为正例，实际为负例
# FN：False Negative，假负例，预测为负例，实际为正例
def TP_TN_FP_FN(confusion_matrix):
    """
    - confusion_matrix : 混淆矩阵
    """
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[i])):
            if i == j:
                TP += confusion_matrix[i][j]
            else:
                FP += confusion_matrix[i][j]
                FN += confusion_matrix[j][i]
    TN = sum(sum(confusion_matrix)) - TP - FP - FN
    return TP, TN, FP, FN