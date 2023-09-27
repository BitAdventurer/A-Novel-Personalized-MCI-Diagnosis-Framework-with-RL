import numpy as np
import config
import torch
import util
from datetime import datetime
from multiprocessing import Pool, freeze_support
import evaluate_best_player_test_p
import seed
args = config.parser.parse_args()

path                    = args.path
best_path               = args.self_data
backup_path             = args.buffer_data
dualnetwork_best_path   = args.dualnetwork_best_path
dualnetwork_best2_path  = args.dualnetwork_best2_path
dualnetwork_target_path = args.dualnetwork_target_path
dualnetwork_target2_path = args.dualnetwork_target2_path
dualnetwork_init_path    = args.dualnetwork_model_init_path
dualnetwork_init2_path    = args.dualnetwork_model_init2_path

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
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

    def __call__(self, val_loss):

        score = val_loss
        print('Score : ', score , 'Best score : ', self.best_score)

        if self.best_score is None:
            self.best_score = score
            return False

        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return False

        else:
            if self.verbose:
                self.trace_func(f'\n Objective function decreased ({self.best_score:.6f} --> {val_loss:.6f}).')
            self.best_score = score
            self.counter = 0

            return True



