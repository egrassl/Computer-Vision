import numpy as np
from matplotlib import pyplot as plt


class Retrieval_Pr(object):

    def __init__(self, precisions, recalls):
        self.precisions = precisions
        self.recalls = recalls

    def plot_curves(self):

        for i in range(0, len(self.recalls)):
            self.plot_curve(self.recalls[i], self.precisions[i], "PR Curve")

    def plot_mean_curve(self):
        mean_recall = np.mean(self.recalls, axis=1)
        mean_precision = np.mean(self.precisions, axis=1)
        self.plot_curve(mean_recall, mean_precision, "PR mean Curve")


    @staticmethod
    def plot_curve(recall, precision, title):
        plt.title(title)
        plt.xlabel = "Recall"
        plt.ylabel = "Precision"
        plt.plot(recall, precision, 'g')
        plt.show()
