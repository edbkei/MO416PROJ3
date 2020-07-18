from sklearn.metrics import accuracy_score, precision_score, recall_score


class RegressorStats:
    @staticmethod
    def get_stats(y_real, y_pred):
        precision = precision_score(y_real, y_pred, average='macro')
        recall = recall_score(y_real, y_pred, average='macro')

        return {
            'accuracy': accuracy_score(y_real, y_pred),
            'precision': precision,
            'recall': recall,
            'f0.5': RegressorStats.f_beta_score(precision, recall, beta=0.5),
            'f1': RegressorStats.f_beta_score(precision, recall, beta=1),
            'f2': RegressorStats.f_beta_score(precision, recall, beta=2)
        }


    @staticmethod
    def f_beta_score(precision, recall, beta=1):
        beta_sq = beta ** 2

        return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


    @staticmethod
    def mean_error(regressors, value):
        sum = 0
        total = 0

        for regressor in regressors:
            sum += regressor[value][regressor['final_iteration']]
            total += 1

        return sum / total
