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
    """
    This function evaluates the agent's performance using a specified model.
    subj: Subject or environment instance to be evaluated.
    iters: Number of iterations.
    """

    # Setting seed for reproducibility.
    seed.seed_everything(args.seed)

    # Initializing data containers and counters.
    input_state_label_set, data_x_label_single = [], []
    count_input = 0
    count_terminal_q, terminal_state_label_set_q = 0, []

    ############### LOAD CLASSIFIER
    # Loading and setting the classifier model to evaluation mode.
    classifier = util.classifier() 
    classifier.eval()

    ############### LOAD ACTOR-CRITIC MODEL
    # Loading the trained Actor-Critic model and setting it to evaluation mode.
    model = torch.load(args.dualnetwork_best_path, map_location='cpu')
    model.eval()

    ############### INIT HISTORY
    # Initializing containers to store state, action, policy, adjacency matrix, reward, episode, and Q-values history.
    state_history, action_history, action_nc_history, action_mci_history, policy_history, adj_history, reward_history, episode_history, q_history \
        = [], [], [], [], [], [], [], [], []

    ############### LOAD TEST DATA
    # Loading test data and labels.
    test_data = util.validation_data(args.split)
    test_label = util.validation_label(args.split)
    data_x = test_data[subj] 
    data_x_label = test_label[subj]  

    # Converting label representation.
    if data_x_label==[1,0]: data_x_label = 0
    elif data_x_label==[0,1]: data_x_label = 1

    ############### INIT ENVIRONMENT
    # Initializing the environment state and adjacency matrix.
    state = disconnection.State(data_x, subj, False)  # validation
    state_history.append(state.piece)
    adj = torch.rand(116,116) 
    adj = adj.new_ones(116,116)  # adj:116,116
    adj_state = disconnection_laplace.State(adj)  # adj_state:116,116
    adj_history.append(adj_state.piece)

    ############### INIT STATE VISUALIZATION
    # Generating and saving initial state heatmap if plotting is enabled.
    if args.plt:    
        dir_name = f'/heatmap/{"NC" if data_x_label == 0 else "MCI"}_val_{subj}'
        os.makedirs(path + dir_name, exist_ok=True)
        data = pd.DataFrame(data_x)
        sns.heatmap(data=data, annot=False, square=True, fmt='.2f', cmap='RdYlBu_r', cbar=False)
        plt.xticks([])
        plt.yticks([])

        plt.savefig(f"{path}{dir_name}/{now.hour:02}{now.minute:02}_init_val_state{subj}.jpg", format='jpg', dpi=100)
        plt.close()

    ############### EVALUATION START
    # Loading hyperparameters and initializing state value list.
    STOP_POINT, TEMPERATURE, _, _, _, _, _, _, _ = util.load_hyperparameter()
    state_value = []
    state_value_ls = []


     # Iterate over the predefined stop point, each iteration representing a time step or an action taken by the agent.
    for j in range(STOP_POINT):
        # Convert the current state to a tensor variable.
        state_array = np.array(state.piece, dtype=float)
        state_variable = torch.Tensor(state_array.reshape(1, 1, 116, 116))

        # Compute policy probabilities from Actor Model and detach it from the graph.
        p = util.PI(model, state_variable, torch.tensor(adj_state.piece), False)
        p = p.detach()

        ############### PREVENT ACTION OVERLAP
        # Apply softmax to policy outputs and enforce action overlap rules.
        p = util.softmax(p[0])
        origin_p = p
        p = util.overlab(action_history, p)
        # Sample an action based on the adjusted policy probabilities.
        action = np.random.choice(state.legal_actions(), p=p)

        ############### COMPUTE Q-VALUE
        # Compute the Q-value of the chosen action.
        q = util.Q(model, state_variable, torch.tensor(np.array([action])).reshape(1, -1), torch.tensor(adj_state.piece), False)

        ############### UPDATE HISTORY
        # Append current action and state to their respective histories.
        action_history.append(action)
        if data_x_label == 0: action_nc_history.append(action)
        else: action_mci_history.append(action)

        ############### UPDATE STATE VALUES
        # Compute and store state values and Q-values.
        if not args.state_value:
            for aa in range(args.action):
                pi_val = util.PI(model, state_variable, torch.tensor(adj_state.piece), False)[0][aa].detach().numpy()
                q_val = util.Q(model, state_variable, torch.tensor(np.array([action])).reshape(1, -1), torch.tensor(adj_state.piece), False).detach().numpy()
                state_value.append(pi_val * q_val)
            state_value_ls.append(sum(state_value))
            q_history.append(state_value_ls[0]) 
        else:
            q_history.append(q.tolist())

        policy_history.append(p[0])

        state_value, state_value_ls = [], []

        ############### COMPUTE NEXT STATE ADJACENCY MATRIX
        # Update state and adjacency matrix.
        state = state.next(action, False)
        adj_state = adj_state.next(action)
        state_history.append(state.piece)
        adj_history.append(adj_state.piece)

        # Compute accuracy of classifier on the current label and state.
        label_acc = classifier(torch.tensor(test_data[subj].reshape(1, 116, 116)).float(), adj.float())
        state_acc = classifier(torch.tensor(state_history[j].reshape(1, 116, 116)).float(), torch.tensor(adj_history[j]).clone().detach().float())

        ############### UPDATE EPISODE SUCCESS INFO
        # Check and update episode success information based on the accuracy of classifier predictions.
        with open(path + f'/terminal/val/val_succes_idx/{subj}.txt', mode='rt', encoding='utf-8') as f:
            lines = f.readlines()
            SUCCES_IDX = float(lines[0])

        label_acc_p, label_acc_label = torch.max(label_acc, 1)
        state_acc_p, state_acc_label = torch.max(state_acc, 1)

        # Update the episode length and rewards history based on whether the classifier's predictions are correct or not.
        episode_len = STOP_POINT - (j + 1)
        reward = args.reward * pow(args.discount_factor, episode_len) * prev_iters  # * state_acc_p.tolist()[0])
        reward_history.append(reward if label_acc_label.tolist() == state_acc_label.tolist() else -reward)


        ############### GENERATE AND SAVE POLICY, PROBABILITY, AND STATE IMAGES
        if args.plt:
            # Create directories to save plots based on the label of the data.
            label_dir = 'NC' if data_x_label == 0 else 'MCI'
            os.makedirs(f'{path}/heatmap/{label_dir}_val_{subj}', exist_ok=True)
            
            # Configure plot dimensions for policy bar graph.
            plt.figure(figsize=(20, 8))
            # Plot a bar graph representing the policy probabilities.
            plt.bar(np.arange(args.action), origin_p, label='p0', width=0.4, color='springgreen', edgecolor='black', linewidth=0)
            plt.legend()
            plt.grid()   
            
            # Save the policy probability bar graph.
            filename = f'{path}/heatmap/{label_dir}_val_{subj}/{now.hour:02}{now.minute:02}_val_state{j}-A_{action}_p.jpg'
            plt.savefig(filename, format='jpg', dpi=100)
            plt.close()
            
            # Convert current state to DataFrame for heatmap plotting.
            data = pd.DataFrame(state.piece)
            
            # Plot a heatmap representing the current state.
            sns.heatmap(data=data, annot=False, fmt='.2f', linewidths=0, cmap='RdYlBu_r', cbar=False)
            plt.tight_layout()
            plt.xticks([])
            plt.yticks([])
            
            # Save the state heatmap.
            filename = f'{path}/heatmap/{label_dir}_val_{subj}/{now.hour:02}{now.minute:02}_val_state{j}-A_{action}.jpg'
            plt.savefig(filename, format='jpg', dpi=100)
            plt.close()

        if action==116:
            break
                ############### DETERMINE TERMINAL STATE VALUE
    Terminal_q_idx = np.argmax(q_history)

    ############### COMPARE INITIAL AND TERMINAL STATE VALUES
    input_acc = classifier(torch.tensor(test_data[subj].reshape(1, 116, 116)).float(), adj.float())
    test_q_adj = torch.tensor(adj_history[Terminal_q_idx]).clone().detach()
    terminal_q_acc = classifier(torch.tensor(state_history[Terminal_q_idx].reshape(1, 116, 116)).float(), test_q_adj.float())
    _, input_state_label = torch.max(input_acc, 1)
    _, terminal_state_q_label = torch.max(terminal_q_acc, 1)

    episode_history.append(Terminal_q_idx)

    ############### SAVE Q HISTORY
    with open(path + f'/result/value_hist/val_q_hist{subj}.txt', mode='at', encoding='utf-8') as f:
        f.writelines(f'{q_history}\n')

    ############### SAVE REWARD HISTORY
    with open(path + f'/result/value_hist/val_reward_hist{subj}.txt', mode='at', encoding='utf-8') as f:
        f.writelines(f'{reward_history}\n')

    # Create directories to store various pieces of information, values, actions and classifier outcomes at each iteration.
    dirs = [
        f'/terminal/val/info/{iters}', 
        f'/terminal/val/value/{iters}', 
        f'/terminal/val/action/{iters}'
    ]
    
    for dir in dirs:
        os.makedirs(path + dir, exist_ok=True)

    ############### SAVE ACTION INFORMATION
    info_file = path + f'/terminal/val/info/{iters}/{subj}.txt'
    value_file = path + f'/terminal/val/value/{iters}/{subj}.txt'
    action_file = path + f'/terminal/val/action/{iters}/{subj}.txt'

    if terminal_state_q_label == data_x_label:
        with open(info_file if Terminal_q_idx != 0 else action_file, mode='at', encoding='utf-8') as f:
            f.writelines(f'{action_history[:(Terminal_q_idx)] if Terminal_q_idx != 0 else []}\n')
        
        if Terminal_q_idx != 0:
            with open(value_file, mode='at', encoding='utf-8') as f:
                f.writelines(f'{q_history[:(Terminal_q_idx)]}\n')
    else:
        with open(action_file, mode='at', encoding='utf-8') as f:
            f.writelines(f'{action_history[:(Terminal_q_idx)] if Terminal_q_idx != 0 else []}\n')
    
    ############### SAVE REWARD DATA
    reward_data = pd.DataFrame([reward_history[Terminal_q_idx]])
    reward_data.to_csv(path + f'/reward/r/val_reward_q_{subj}.csv', index=False, mode='a')

    ############### DISPLAY EVALUATION RESULTS
    print(f'*************** Validation Evaluation {subj + 1}/{len(test_data)} ***************')
    print('Episode length : ', STOP_POINT)
    print('Terminal_idx : ', Terminal_q_idx)
    print('Input acc : ', input_acc[0].tolist(), 'Q Terminal acc : ', terminal_q_acc[0].tolist())
    print('Validation label : ', data_x_label)
    print("Action history : ", action_history)
    print("Q history : ", q_history)

    ############### SAVE RESULTS
    if input_state_label == data_x_label: 
        count_input += 1
    if terminal_state_q_label == data_x_label: 
        count_terminal_q += 1


    with open(path + '/result/val_best_test.txt', mode='at', encoding='utf-8') as f:
      f.writelines(f'********Val {subj+1}/{len(test_data)}********\n')
      f.writelines(f'Episode length : {STOP_POINT}\n')
      f.writelines(f'Teminal idx : {Terminal_q_idx}\n')
      f.writelines(f'Input acc : {classifier(torch.tensor(data_x.reshape(1, 116, 116)).float(), adj.clone().detach().float())}\n')

      for i, (state_n, adj_n) in enumerate(zip(state_history, adj_history)):
          state_n_acc = classifier(torch.tensor(state_n.reshape(1, 116, 116)).float(), torch.tensor(adj_n).clone().detach().float())
          f.writelines(f'State {i} predict : {state_n_acc[0]}\n')
      
      f.writelines(f'Terminal predict : {classifier(torch.tensor(state_history[Terminal_q_idx].reshape(1, 116, 116)).float(), torch.tensor(adj_history[Terminal_q_idx]).clone().detach().float())[0]}\n')
      f.writelines(f'Label : {data_x_label}\n')
      f.writelines(f'Action history : {action_history}\n')
      f.writelines(f'Q history : {q_history}\n')

    input_state_label_set.extend(input_state_label.tolist())
    terminal_state_label_set_q.extend(terminal_state_q_label.tolist())
    data_x_label_single.extend([data_x_label])
  
    np.save(path+f'/result/info_val_p/val_{subj}_input', input_state_label_set)
    np.save(path+f'/result/info_val_p/val_{subj}_terminal_q', np.array(terminal_state_label_set_q))
    np.save(path+f'/result/info_val_p/val_{subj}_label', data_x_label_single)
    np.save(path+f'/result/info_val_p/val_{subj}_count_input', [count_input])
    np.save(path+f'/result/info_val_p/val_{subj}_count_terminal_q', [count_terminal_q])
  
    ############### SUCCES INFO SAVE
    success_idx_file = path + f'/terminal/val/val_succes_idx/{subj}.txt'
    with open(success_idx_file, mode='wt', encoding='utf-8') as f:
        f.writelines(f'{1.0 if terminal_state_q_label == data_x_label else -1.0}')

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

    sensitivity = sum([sens * weight for sens, weight in zip(sensitivities, weights)]) / total_samples
    specificity = sum([spec * weight for spec, weight in zip(specificities, weights)]) / total_samples
    precision = sum([prec * weight for prec, weight in zip(precisions, weights)]) / total_samples
    f1 = sum([f1 * weight for f1, weight in zip(f1_scores, weights)]) / total_samples

    metrics = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1
    }

    return sensitivity, specificity, precision, f1

def confusion(iters, break_point=False):
    test_data = util.validation_data(args.split)
    confusion_input, confusion_label, confusion_terminal_q = [], [], []
    confusion_count_input, confusion_count_terminal_q = 0, 0

    for i in tqdm(range(len(test_data))):
        confusion_input.extend(np.load(path+f'/result/info_val_p/val_{i}_input.npy'))

        confusion_terminal_q.extend(np.load(path+f'/result/info_val_p/val_{i}_terminal_q.npy'))

        confusion_label.extend(np.load(path+f'/result/info_val_p/val_{i}_label.npy'))
        confusion_count_input+=(np.load(path+f'/result/info_val_p/val_{i}_count_input.npy'))

        confusion_count_terminal_q+=(np.load(path+f'/result/info_val_p/val_{i}_count_terminal_q.npy'))

    target_names = ['NC', 'MCI']
    print(classification_report(confusion_label , confusion_input, target_names=target_names))

    init_matrix = confusion_matrix(confusion_label , confusion_input)
    init_weighted_sensitivity, init_weighted_specificity, init_weighted_precision, init_weighted_f1 = calculate_metrics(init_matrix)

    terminal_matrix = confusion_matrix(confusion_label , confusion_terminal_q)
    terminal_weighted_sensitivity, terminal_weighted_specificity, terminal_weighted_precision, terminal_weighted_f1 = calculate_metrics(terminal_matrix)

    if break_point:
        with open(path + f'/result/info_result/result_{iters}.txt', mode='at', encoding='utf-8') as f:
            f.writelines('****** Validation Best P Result... ******\n')
            f.writelines(f'P INPUT ACC : {confusion_count_input/len(test_data)}\n')
            f.writelines(f'P Terminal ACC : {confusion_count_terminal_q/len(test_data)}\n')
            f.writelines(f'DELTA ACC : {confusion_count_terminal_q/len(test_data) - confusion_count_input/len(test_data)}\n')

            f.writelines(f'Input confusion matrix : \n{init_matrix}\n')
            f.writelines(f'Input precision_score : {init_weighted_precision}\n')
            f.writelines(f'Input recall(sensitivity) : {init_weighted_sensitivity}\n')
            f.writelines(f'Input specificity : {init_weighted_specificity}\n')
            f.writelines(f'Input f1_score : {init_weighted_f1}\n')

            f.writelines(f'Terminal confusion matrix : \n{terminal_matrix}\n')
            f.writelines(f'Terminal precision_score : {terminal_weighted_precision}\n')
            f.writelines(f'Terminal recall(sensitivity) : {terminal_weighted_sensitivity}\n')
            f.writelines(f'Terminal specificity : {terminal_weighted_specificity}\n')
            f.writelines(f'Terminal f1_score : {terminal_weighted_f1}\n')

        ############### Q
        with open(path + '/result/confusion_info/val_acc_plot_q.txt', mode='at', encoding='utf-8') as f:
            f.writelines(f'{confusion_count_terminal_q/len(test_data) - confusion_count_input/len(test_data)}\n')

        with open(path + '/result/confusion_info/val_sensitivity_plot_q.txt', mode='at', encoding='utf-8') as f:
            f.writelines(f'{terminal_weighted_sensitivity-init_weighted_sensitivity}\n')

        with open(path + '/result/confusion_info/val_specificy_plot_q.txt', mode='at', encoding='utf-8') as f:
            f.writelines(f'{terminal_weighted_specificity - init_weighted_specificity}\n')

        with open(path + '/result/confusion_info/val_f1_plot_q.txt', mode='at', encoding='utf-8') as f:
            f.writelines(f'{terminal_weighted_f1 - init_weighted_f1}\n')

        wandb.init(project=args.new_server, name='val', allow_val_change=True)
        wandb.log({
        "Val ACC Delta": confusion_count_terminal_q/len(test_data) - confusion_count_input/len(test_data),
        "Val sensitivity": terminal_weighted_sensitivity - init_weighted_sensitivity,
        "Val specificy": terminal_weighted_specificity - init_weighted_specificity,
        "Val F1": terminal_weighted_f1 - init_weighted_f1
        })

    if iters > 0:
        ############### league agent
        agent_history = []
        with open(path +f'/result/confusion_info/val_acc_plot_q.txt', mode='rt', encoding='utf-8') as f:
            lines = f.readlines()  
            current_agnet = float(lines[-1])
            prev_agent = float(lines[-2])
            for line in lines:
                agent_history.append(float(line))
                
            strong_agent = max(agent_history)

        if current_agnet < strong_agent:
            return confusion_count_terminal_q/len(test_data), False

        else:
            return confusion_count_terminal_q/len(test_data), True
    else:
        return confusion_count_terminal_q/len(test_data), False

