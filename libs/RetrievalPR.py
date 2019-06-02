import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


class Retrieval_Pr(object):

    def __init__(self, precisions, recalls, labels):
        self.precisions = precisions
        self.recalls = recalls
        self.labels = labels

    def plot_curves(self):

        for i in range(0, len(self.recalls)):
            self.plot_curve(self.recalls[i], self.precisions[i], "PR Curve for class " + self.labels[i])
            #x_new, y_new = self.fitcurve(self.recalls[i], self.precisions[i])
            #self.plot_curve(x_new, y_new, "PR Curve for class FITTED" + self.labels[i])

    @staticmethod
    def fitcurve(x, y):
        z = np.polyfit(x, y, 8)
        f = np.poly1d(z)
        x_new = np.linspace(x[0], x[-1], 50)
        y_new = f(x_new)
        x_new[0] = 0.0
        y_new[0] = 1.0
        return x_new, y_new

    def get_mean_curve(self):
        fit_precisions = []
        f_recall = []
        for i in range(0, len(self.precisions)):
            f_recall, fit_precision = self.fitcurve(self.recalls[i], self.precisions[i])
            fit_precisions.append(fit_precision)

        fit_precisions = np.array(fit_precisions)

        mean_precision = np.empty(50)
        for i in range(0, 50):
            mean = 0.0
            for j in range(0, len(fit_precisions)):
                mean += fit_precisions[j][i]
            mean /= 11.0
            mean_precision[i] = mean

        return f_recall, mean_precision

    def plot_mean_curve(self):
        mean_recall, mean_precision = self.get_mean_curve()
        self.plot_curve(mean_recall, mean_precision, "PR mean Curve")


    @staticmethod
    def plot_curve(recall, precision, title):
        plt.title(title)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.plot(recall, precision, 'g')
        plt.show()
