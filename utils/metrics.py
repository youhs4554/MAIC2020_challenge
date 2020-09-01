from sklearn import metrics
from sklearn.metrics import average_precision_score


def convert_to_numpy(fn):
    def wrapper(*args):
        y_true, y_pred = args
        return fn(y_true.cpu().numpy(), y_pred.detach().cpu().numpy())
    return wrapper


def compute_accuracy(y_true, y_pred):
    return y_pred.ge(0.5).eq(y_true).float().mean().item()


@convert_to_numpy
def compute_auc(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(
        y_true, y_pred, pos_label=1)
    score = metrics.auc(fpr, tpr)

    return score


@convert_to_numpy
def compute_auprc(y_true, y_pred):
    score = average_precision_score(y_true, y_pred)
    return score
