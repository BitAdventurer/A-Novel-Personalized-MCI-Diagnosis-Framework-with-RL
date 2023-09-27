import os
import random
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from tqdm import tqdm

import disconnection
import disconnection_laplace
import gcn_util
import seed
import util
import wandb
from DualNetwork import ActorCritic
from General_baseline import chev

import pyximport; pyximport.install()

warnings.filterwarnings("ignore") 
import config

args  = config.parser.parse_args()
now   = datetime.now()
jeilt = args.jeilt
path  = args.path

############### EVALUATE AGENT
def evaluate_best_player(subj, iters):
    seed.seed_everything(args.seed)
    input_state_label_set, data_x_label_single = [], []
    count_input = 0
    count_terminal_q, terminal_state_label_set_q = 0, []

    ############### CLASSIFIER LOAD
    classifier = util.classifier() 
    classifier.eval()

    ############### ACTOR-CRITIC LOAD
    model = torch.load(args.dualnetwork_model_init_path, map_location='cpu')
    model.eval()

    ############### HISTORY
    state_history, action_history, action_nc_history, action_mci_history, policy_history, adj_history, reward_history, episode_history, q_history \
        = [], [], [], [], [], [], [], [], []

    ############### DATA LOAD
    test_data = util.validation_data(args.split)
    test_label = util.validation_label(args.split)
    data_x = test_data[subj] 
    data_x_label = test_label[subj]  

    if data_x_label==[1,0]: data_x_label = 0
    elif data_x_label==[0,1]: data_x_label = 1



    ############### ENVIROMENT INIT
    state = disconnection.State(data_x, subj, False) # validation
    state_history.append(state.piece)
    adj = torch.rand(116,116) 
    adj = adj.new_ones(116,116) #adj:116,116
    adj_state = disconnection_laplace.State(adj) #adj_state:116,116
    adj_history.append(adj_state.piece)

    ############### START
    STOP_POINT, TEMPERATURE, _, _, _, _, _, _, _ = util.load_hyperparameter()
    state_value=[]
    state_value_ls=[]

    for j in range(STOP_POINT):
        state_array = np.array(state.piece, dtype=float)
        state_variable = torch.Tensor(state_array.reshape(1, 1, 116, 116))
        p = util.PI(model, state_variable, torch.tensor(adj_state.piece), False)
        p = p.detach()

        ############### ACTION OVERLAP PREVENT    
        p = util.softmax(p[0])
        origin_p = p
        p = util.overlab(action_history, p)
        action = np.random.choice(state.legal_actions(), p=p)

        ############### Q
        q = util.Q(model, state_variable, torch.tensor(np.array([action])).reshape(1,-1), torch.tensor(adj_state.piece), False)

        ############### INIT HISTORY APPEND
        action_history.append(action)

        ############### MCI/NC ACTION APPEND
        if data_x_label == 0: action_nc_history.append(action)
        else: action_mci_history.append(action)

        ############### VALUE APPEND
        q_history.append(q.tolist()) 

        policy_history.append(p[0])

        state_value, state_value_ls = [], []

        ############### GET NEXT STATE ADJ
        state = state.next(action, False)
        adj_state = adj_state.next(action)
        state_history.append(state.piece)
        adj_history.append(adj_state.piece)

        if action==116:
            break
    ############### TERMINAL STATE VALUE
    Terminal_q_idx = np.argmax(q_history)


    ############### S0, TERMINAL COMP
    input_acc = classifier(torch.tensor(test_data[subj].reshape(1, 116, 116)).float(), adj.float())
    test_q_adj = torch.tensor(adj_history[Terminal_q_idx]).clone().detach()
    terminal_q_acc = classifier(torch.tensor(state_history[Terminal_q_idx].reshape(1, 116, 116)).float(), test_q_adj.float())
    _, input_state_label = torch.max(input_acc, 1)
    _, terminal_state_q_label = torch.max(terminal_q_acc, 1)


    ############### RESLUT SAVE
    if input_state_label == data_x_label: count_input += 1
    if terminal_state_q_label == data_x_label: count_terminal_q += 1


    input_state_label_set.extend(input_state_label.tolist())
    terminal_state_label_set_q.extend(terminal_state_q_label.tolist())

    if data_x_label == 0: data_x_label_single.extend([0])
    elif data_x_label == 1: data_x_label_single.extend([1])
    elif data_x_label == 2: data_x_label_single.extend([2])
    elif data_x_label == 3: data_x_label_single.extend([3])

    np.save(path+f'/result/info_val_c/{subj}_input', input_state_label_set)

    np.save(path+f'/result/info_val_c/{subj}_terminal_q', np.array(terminal_state_label_set_q))

    np.save(path+f'/result/info_val_c/{subj}_label', data_x_label_single)

    np.save(path+f'/result/info_val_c/{subj}_count_input', np.array(count_input))

    np.save(path+f'/result/info_val_c/{subj}_count_terminal_q', np.array(count_terminal_q))

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

def confusion():
    test_data = util.validation_data(args.split)
    confusion_input, confusion_label, confusion_terminal_q = [], [], []
    confusion_count_input, confusion_count_terminal_q = 0, 0

    for i in tqdm(range(len(test_data))):
        confusion_input.extend(np.load(path+f'/result/info_val_c/{i}_input.npy'))

        confusion_terminal_q.extend(np.load(path+f'/result/info_val_c/{i}_terminal_q.npy'))

        confusion_label.extend(np.load(path+f'/result/info_val_c/{i}_label.npy'))
        confusion_count_input+=(np.load(path+f'/result/info_val_c/{i}_count_input.npy'))

        confusion_count_terminal_q+=(np.load(path+f'/result/info_val_c/{i}_count_terminal_q.npy'))


    init_matrix = confusion_matrix(confusion_label , confusion_input)
    init_weighted_sensitivity, init_weighted_specificity, init_weighted_precision, init_weighted_f1 = calculate_metrics(init_matrix)

    terminal_matrix = confusion_matrix(confusion_label , confusion_terminal_q)
    terminal_weighted_sensitivity, terminal_weighted_specificity, terminal_weighted_precision, terminal_weighted_f1 = calculate_metrics(terminal_matrix)



    ############### Q
    with open(path + '/result/confusion_info/val_acc_plot_q_c.txt', mode='at', encoding='utf-8') as f:
        f.writelines(f'{confusion_count_terminal_q/len(test_data) - confusion_count_input/len(test_data)}\n')

    with open(path + '/result/confusion_info/val_sensitivity_plot_q_c.txt', mode='at', encoding='utf-8') as f:
        f.writelines(f'{terminal_weighted_sensitivity-init_weighted_sensitivity}\n')

    with open(path + '/result/confusion_info/val_specificy_plot_q_c.txt', mode='at', encoding='utf-8') as f:
        f.writelines(f'{terminal_weighted_specificity - init_weighted_specificity}\n')

    with open(path + '/result/confusion_info/val_f1_plot_q_c.txt', mode='at', encoding='utf-8') as f:
        f.writelines(f'{terminal_weighted_f1 - init_weighted_f1}\n')



    return confusion_count_terminal_q/len(test_data)


