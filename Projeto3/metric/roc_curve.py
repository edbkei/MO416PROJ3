from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt


class ROCCurve:
    @staticmethod
    def plot_roc_curve(y_real, y_pred, y_pred_prob):
        logit_roc_auc = roc_auc_score(y_real, y_pred)
        fpr, tpr, thresholds = roc_curve(y_real, y_pred_prob[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('Log_ROC')
        plt.show()