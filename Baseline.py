import os
import ast
import pyximport; pyximport.install()

# Third-party imports
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    precision_score, 
    recall_score, 
    roc_auc_score, 
    roc_curve
)
from torch.autograd import Variable

# Local module imports
import config
import disconnection
import disconnection_laplace
import gcn_util
import seed
import util
from General_baseline import chev

# Parse Arguments
args = config.parser.parse_args()
jeilt = args.jeilt
path = args.path

# Setup
use_cuda = torch.cuda.is_available()
torch.set_num_threads(os.cpu_count() * 2)


def test(x, adj, classifier):
    """
    Test the given classifier with input x and adjacency matrix adj.
    
    :param x: Input tensor.
    :param adj: Adjacency matrix.
    :param classifier: The classifier to be tested.
    :return: The classifier’s output data.
    """
    with torch.no_grad():
        output = classifier(x, adj)
        
    return output.data


if __name__ == "__main__":
    # You can put any code here that should run when the script is executed,
    # for example, a call to the `test` function with appropriate parameters.
    pass
