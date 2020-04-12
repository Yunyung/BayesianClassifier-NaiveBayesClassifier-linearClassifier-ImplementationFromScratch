import numpy as np
import matplotlib.pyplot as plt


class Evaluation:
    """
        Evaluation Module
    """

    def __init__(self):
        pass

    def confusion_matrix(self, y_true, y_pred):
        """
            Calculate  class confusion Matrix to evaluation classifier. Can apply to multi classes dataset
            Note: This function assume that label from 0 to N(The Maximim of label value)
            args:
                y_true: Ground truth (correct) target values.
                y_pred: Estimated targets as returned by a classifier

            returns:
                Confusion matrix: narray of shape:(n_classs, n_classes)
        """
        # find the maximun of label value
        max_label_value = np.max(y_true)

        print("max_label_value:", max_label_value)
        confusionMatrix = np.zeros((max_label_value + 1, max_label_value + 1))
        print(confusionMatrix)
        # buidl confusion Matrix
        for i in range(len(y_true)):
            confusionMatrix[y_true[i], y_pred[i]] += 1

        return confusionMatrix

    def roc_curve(self, y_true, y_score):
        """
            Plot roc_curve to evaluation classifier
            Note: this implementation is restricted to the binary classification task.

            args:
                y_true: True binary labels. If labels are not either {-1, 1} or {0, 1}, then pos_label should be explicitly given.
                y_score: Target scores, can either be probability estimates of the positive class

            return:
                fpr : Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i].
                tpr : Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
                threshold : Decreasing thresholds on the decision function used to compute fpr and tpr.
        """

        tpr_list = []
        fpt_list = []
        thresholds = np.linspace(1.1, 0, 10)

        for t in thresholds:
            y_pred = np.zeros(y_true.shape[0])
            # print(y_score)
            y_pred[y_score >= t] = 1
            TP = y_pred[(y_pred == y_true) & (y_true == 1)].shape[0]
            TN = y_pred[(y_pred == y_true) & (y_true == 0)].shape[0]
            FN = y_pred[(y_pred != y_true) & (y_true == 1)].shape[0]
            FP = y_pred[(y_pred != y_true) & (y_true == 0)].shape[0]
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)
            tpr_list.append(TPR)
            fpt_list.append(FPR)

        return tpr_list, fpt_list, thresholds

    def plot_roc_curve(self, y_true, y_score):
        """
            plot roc curve 
        """
        tpr_list, fpt_list, thresholds = self.roc_curve(y_true, y_score)

        plt.plot(fpt_list, tpr_list, 'b')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()

    def roc_auc_score(self, y_true, y_score):
        """
            Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
            Note: this implementation can be used with binary,
        """
        tpr_list, fpt_list, thresholds = self.roc_curve(y_true, y_score)

        print(tpr_list)
        print(fpt_list)
        score = np.zeros(1)
        for i in range(len(tpr_list) - 1):
            score += (fpt_list[i + 1] - fpt_list[i]) * (tpr_list[i + 1])
        return score
