import numpy as np
import torch
import os
import time
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score,
    roc_auc_score, roc_curve, classification_report)
import config
args = config.parser.parse_args()

path = args.path

def calculate_metrics(matrix):
    num_classes = matrix.shape[0]

    sensitivities = []
    specificities = []
    precisions = []
    f1_scores = []

    for i in range(num_classes):
        TP = matrix[i, i]
        FN = sum(matrix[i, :]) - TP
        FP = sum(matrix[:, i]) - TP
        TN = np.sum(matrix) - TP - FN - FP

        # Sensitivity (True Positive Rate)
        sensitivity = TP / (TP + FN) if TP + FN != 0 else 0
        sensitivities.append(sensitivity)

        # Specificity (True Negative Rate)
        specificity = TN / (TN + FP) if TN + FP != 0 else 0
        specificities.append(specificity)

        # Precision
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        precisions.append(precision)

        # F1 score
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if precision + sensitivity != 0 else 0
        f1_scores.append(f1)

    # Weighted metrics
    total_samples = np.sum(matrix)
    weights = [sum(row) for row in matrix]

    weighted_sensitivity = sum([sens * weight for sens, weight in zip(sensitivities, weights)]) / total_samples
    weighted_specificity = sum([spec * weight for spec, weight in zip(specificities, weights)]) / total_samples
    weighted_precision = sum([prec * weight for prec, weight in zip(precisions, weights)]) / total_samples
    weighted_f1 = sum([f1 * weight for f1, weight in zip(f1_scores, weights)]) / total_samples

    metrics = {
        'sensitivity': weighted_sensitivity,
        'specificity': weighted_specificity,
        'precision': weighted_precision,
        'f1': weighted_f1
    }

    return weighted_sensitivity, weighted_specificity, weighted_precision, weighted_f1

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, val_cm1=0, val_cm2=0, test_cm1=0, test_cm2=0, split=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.val_cm1 = val_cm1
        self.val_cm2 = val_cm2

        

    def __call__(self, val_loss, model, val_cm1, val_cm2, split):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, val_cm1, val_cm2, split)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, val_cm1, val_cm2, split)
            self.counter = 0




    def save_checkpoint(self, val_loss, model, val_cm1, val_cm2, split):

        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'\n Validation score SOTA ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if os.path.isfile(path+f'/baseline/fold{args.fold}/checkpoint/'+ f'{split}' + self.path):
            os.remove(path+f'/baseline/fold{args.fold}/checkpoint/'+ f'{split}' + self.path)

        torch.save(model.eval(), path+f'/baseline/fold{args.fold}/checkpoint/'+ f'{split}' + self.path )

        self.val_loss_min = val_loss
