from sklearn import metrics
from sklearn.metrics import average_precision_score


def compute_accuracy(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred)


def compute_auc(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(
        y_true, y_score, pos_label=1)
    score = metrics.auc(fpr, tpr)

    return score


def compute_auprc(y_true, y_score):
    score = average_precision_score(y_true, y_score)
    return score
