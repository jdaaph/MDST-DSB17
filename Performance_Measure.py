from __future__ import print_function
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

perc = 10**-5

class performance_classification():

    def __init__(self):
        pass

    def prepare_data(self, p):
        return np.maximum(np.minimum(p, 1.0-perc), perc)

    def auc_measure(self, y_true, y_pred):
        return roc_auc_score(y_true, y_pred)

    def logloss_measure(self, y_true, y_pred):
        n = float(len(y_true))
        logloss = y_true * np.log10(y_pred) +\
                  (1.0 - y_true) * np.log10(1.0 - y_pred)
        return - np.sum(logloss) / n

    def brier_score_loss_measure(self, y_true, y_pred):
        """
        The calibration performance is evaluated with Brier score, reported in the legend (the smaller the better)
        :param y_true: true class
        :param y_pred: predicted probability
        :return:
        """
        return brier_score_loss(y_true, y_pred)

    def plot_roc_curve(self, y_true, y_pred, fname='plot', fdir='./plots/', interactive=False):

        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        if ~interactive:
            import matplotlib
            matplotlib.use('Agg')

        if not os.path.exists(fdir):
            os.makedirs(fdir)

        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        # Plot of a ROC curve for a specific class
        plt.clf()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc=4)
        plt.savefig(fdir+fname+'_roc_curve.png', bbox_incehs='tight')

    def measures(self, y_true, y_pred, verbose=True, plot=True, fname='none', fdir='./plots/'):
        """
        calculate some performance measurements
        :param y_true: true class
        :param y_pred: predicted probability
        :param verbose: If TRUE it prints out logloss, AUC, Brier score
        :param verbose: If TRUE it plot ROC curve
        :return: logloss, AUC, Brier score
        """

        y_pred_star = self.prepare_data(np.array(y_pred))
        y_true_star = np.array(y_true)

        auc_out = self.auc_measure(y_true_star, y_pred_star)
        logloss_out = self.logloss_measure(y_true_star, y_pred_star)
        brier_score_out = self.brier_score_loss_measure(y_true_star, y_pred_star)

        if verbose:
            if logloss_out < 0.05:
                print('logloss: %0.4f , auc: %0.2f , brier score: %0.2f'
                      %(logloss_out, auc_out, brier_score_out))
            else:
                print('logloss: %0.2f , auc: %0.2f , brier score: %0.2f'
                      %(logloss_out, auc_out, brier_score_out))

        if plot:
            self.plot_roc_curve(y_true_star, y_pred_star, fname=fname, fdir=fdir)

        return logloss_out, auc_out, brier_score_out

