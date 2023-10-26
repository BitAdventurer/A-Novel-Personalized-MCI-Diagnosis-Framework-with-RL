import ast
import copy
import gc
import math
import os
import pathlib
import time
from cmath import log10
from collections import Counter, defaultdict
from datetime import datetime
from multiprocessing import Pool, freeze_support
from pathlib import Path
from pprint import pprint

import disconnection
import disconnection_laplace
import gcn_util
import graphviz
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from tqdm import tqdm
import evaluate_best_player_val_p
import config
import reward
import seed
from Baseline import chev

args = config.parser.parse_args()
seed.seed_everything(args.seed)  
path=args.path

def hyperparameter(
        stop_point, temperature, sp_game_count, 
        lr, wd, momentum, process_num, 
        batch_size, buffer_size):
    """
    Writes the provided hyperparameters to a file.

    Args:
        stop_point (int): The stopping point for the training.
        temperature (float): The temperature parameter.
        sp_game_count (int): The game count parameter for self-play.
        lr (float): The learning rate.
        wd (float): The weight decay.
        momentum (float): The momentum.
        process_num (int): The number of processes.
        batch_size (int): The size of the batch.
        buffer_size (int): The size of the buffer.
    """
    
    hyperparameters = [
        stop_point, temperature, sp_game_count,
        lr, wd, momentum, process_num,
        batch_size, buffer_size
    ]

    file_path = os.path.join(path, 'hyperparameter.txt')
    with open(file_path, 'wt', encoding='utf-8') as file:
        file.write('\n'.join(map(str, hyperparameters)))

def load_hyperparameter():
    """
    Loads hyperparameters from a file and returns them as a tuple.

    Returns:
        tuple: A tuple containing the loaded hyperparameters:
            - stop_point (int): The stopping point for the training.
            - temperature (float): The temperature parameter.
            - sp_game_count (int): The game count parameter for self-play.
            - lr (float): The learning rate.
            - wd (float): The weight decay.
            - momentum (float): The momentum.
            - soft_ratio (float): The soft ratio.
            - batch_size (int): The size of the batch.
            - buffer_size (int): The size of the buffer.
    """
    file_path = os.path.join(path, 'hyperparameter.txt')
    
    with open(file_path, 'rt', encoding='utf-8') as file:
        lines = file.readlines()
        
    converters = [int, float, int, float, float, float, float, int, int]
    hyperparameters = tuple(converter(line) for converter, line in zip(converters, lines))
    
    return hyperparameters

def soft_update(source_model, target_model, tau, is_primary_model):
    """
    Performs a soft update on the target model's parameters based on the source model's parameters.
    
    Args:
        source_model (nn.Module): The source model from which the parameters are copied.
        target_model (nn.Module): The target model whose parameters are to be updated.
        tau (float): The soft update mixing factor.
        is_primary_model (bool): Flag indicating if the target model is the primary model.
        
    Returns:
        None
    """
    for target_param, source_param in zip(target_model.state_dict().values(), source_model.state_dict().values()):
        target_param.data.copy_(target_param.data * tau + source_param.data * (1.0 - tau))
        
    model_path = args.dualnetwork_model_init_path if is_primary_model else args.dualnetwork_model_init2_path
    best_path = args.dualnetwork_best_path if is_primary_model else args.dualnetwork_best2_path
    
    target_model = target_model.cuda()
    torch.save(target_model, model_path)
    torch.save(target_model, best_path)


def hard_update(source_model, target_model, is_primary_model):
    """
    Performs a hard update by directly copying the parameters from the source model to the target model.
    
    Args:
        source_model (nn.Module): The source model from which the parameters are copied.
        target_model (nn.Module): The target model whose parameters are to be updated.
        is_primary_model (bool): Flag indicating if the target model is the primary model.
        
    Returns:
        None
    """
    best_path = args.dualnetwork_best_path if is_primary_model else args.dualnetwork_best2_path
    torch.save(target_model, best_path)

from pathlib import Path

def load_data(directory_path, index):
    """
    Load training data based on a directory path and an index.
    
    Args:
        directory_path (str): Path to the directory containing training data files.
        index (int): Index of the training data file to load.
        
    Returns:
        object: Loaded training data.
    """
    history_file_path = sorted(Path(directory_path).glob('*.history'))[index]
    with history_file_path.open(mode='rb') as f:
        return torch.load(f)

def self_load_data(directory_path, file_pattern):
    """
    Load self-play data based on a directory path and a file pattern.
    
    Args:
        directory_path (str): Path to the directory containing self-play data files.
        file_pattern (str): Pattern to match the required self-play data file.
        
    Returns:
        object: Loaded self-play data.
    """
    history_file_path = sorted(Path(directory_path).glob(file_pattern))[0]
    with history_file_path.open(mode='rb') as f:
        return torch.load(f)

############### PLOT
def pi_p0(p0, x, i):
    plt.figure(figsize=(20, 8))
    now = datetime.now()
    os.makedirs(path +'/result/pi_p0/subject{}'.format(x), exist_ok=True)
    plt.figure(figsize=(20, 10))
    plt.bar(np.arange(args.action), p0, label='P', color='springgreen', edgecolor='black')
    plt.legend()
    plt.grid()
    plt.savefig(path+'/result/pi_p0/subject{}/best_pi_p0_{}_{}_{:04}{:02}{:02}{:02}{:02}{:02}.png'.format(x, x, i, now.year, now.month, now.day, now.hour, now.minute, now.second), format='png', dpi=100)
    plt.close()
    gc.collect()

############### SELFPLAY DATA WRITE
def best_write_data(history, subj, start, itx):
    print('Subject {}, {}th write! {}s'.format(itx, subj, round(time.time()-start, 1)))    
    now = datetime.now()
    if args.replay_buffer:
        path_backup = path+'/self_play_backup/{:04}{:02}{:02}{:02}{:02}{:02}_best{}_{}.history'.format(now.year, now.month, now.day, now.hour, now.minute, now.second, itx, subj)
        with open(path_backup, mode='wb') as f:
            torch.save(history, f)

    path2 = path + f'/self_play_best_data/{itx}_{subj}.history'
    with open(path2, mode='wb') as f:
        torch.save(history, f)


def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]


def overlab(action_history, probabilities):
    """
    Adjust the given probabilities based on action history. 
    All probabilities of actions in the history (except action 116) are set to zero.
    Then, the action with the highest probability is selected.

    Args:
        action_history (list): List of actions taken in the past.
        probabilities (list or numpy array): The original probabilities.

    Returns:
        numpy.array: Adjusted probabilities with one action selected.
    """
    
    adjusted_probabilities = probabilities.copy()
    for action in action_history:
        if action != 116:
            adjusted_probabilities[action] = 0

    # Select the action with the highest adjusted probability
    best_action = np.argmax(adjusted_probabilities)
    scores = np.zeros_like(adjusted_probabilities)
    scores[best_action] = 1

    return scores


################ LOAD VALIDATION DATA
def train_data(idx): # 
    return self_load_data(path+f'/data/fold{args.fold}', f'train_fold{args.fold}_split{idx}.history')

################ LOAD VALIDATION LABEL
def train_label(idx): # 
        return self_load_data(path+f'/data/fold{args.fold}', f'train_label_fold{args.fold}_split{idx}.history')

################ LOAD VALIDATION DATA
def validation_data(idx): # 
    return self_load_data(path+f'/data/fold{args.fold}', f'validation_fold{args.fold}_split{idx}.history')

################ LOAD VALIDATION LABEL
def validation_label(idx): # 
        return self_load_data(path+f'/data/fold{args.fold}', f'validation_label_fold{args.fold}_split{idx}.history')

################ LOAD TEST DATA
def test_data(): # 
        return self_load_data(path+f'/data/fold{args.fold}', f'test_fold{args.fold}.history')

################ LOAD TEST LABEL
def test_label():
        return self_load_data(path+f'/data/fold{args.fold}', f'test_label_fold{args.fold}.history')


def softmax(tensor):
    """
    Compute the softmax of a tensor.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: A tensor representing the softmax of the input tensor.
    """
    max_val = tensor.max()  # find the maximum value in the tensor
    exp_tensor = torch.exp(tensor - max_val)  # exponentiate each element and stabilize
    softmax_tensor = exp_tensor / exp_tensor.sum()  # normalize the exponentiated tensor
    return softmax_tensor

def get_device():
    """Returns the appropriate device (CPU or CUDA)"""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Q(model, s, a, m, train_mode):
    """
    Load the critic's output for given inputs.

    Args:
        model: The neural network model.
        s: State tensor.
        a: Action tensor.
        m: Additional model-specific tensor.
        train_mode: If True, sets model to training mode. Otherwise, evaluation mode.

    Returns:
        Tensor: Critic's output.
    """
    model = model.to(get_device())
    model.train() if train_mode else model.eval()
    return model.Q(s.float(), a.float(), m, train_mode).squeeze()

def PI(model, s, m, train_mode):
    """
    Load the actor's output for given inputs.

    Args:
        model: The neural network model.
        s: State tensor.
        m: Additional model-specific tensor.
        train_mode: If True, sets model to training mode. Otherwise, evaluation mode.

    Returns:
        Tensor: Actor's output.
    """
    model = model.to(get_device())
    model.train() if train_mode else model.eval()
    return model.pi(s, m, train_mode, softmax_dim=1)

def T(model, s, adj, train_mode):
    """
    Load the temperature output for given inputs.

    Args:
        model: The neural network model.
        s: State tensor.
        adj: Adjacency tensor.
        train_mode: If True, sets model to training mode. Otherwise, evaluation mode.

    Returns:
        Tensor: Temperature output.
    """
    model = model.to(get_device())
    model.train() if train_mode else model.eval()
    return model.temp(s.float(), adj.float()).squeeze()

def load_classifier():
    """Load the classifier model from the specified path."""
    return torch.load(f'{path}/baseline/fold{args.fold}/checkpoint/{args.split}checkpoint.pt', map_location='cpu')


def target_networks(source_model1, source_model2, target_model1, target_model2):
    """
    Perform a soft update on target networks using parameters from source networks.
    
    Args:
        source_model1 (torch.nn.Module): First source model.
        source_model2 (torch.nn.Module): Second source model.
        target_model1 (torch.nn.Module): First target model to be updated.
        target_model2 (torch.nn.Module): Second target model to be updated.
    """
    
    print('----- Updating Target Networks -----')
    
    # Update target_model1 parameters based on source_model1
    for target_param, source_param in zip(target_model1.parameters(), source_model1.parameters()):
        update_value = source_param.data * args.target_soft_update_ratio 
        target_param.data.copy_(update_value + target_param.data * (1.0 - args.target_soft_update_ratio))
    
    torch.save(target_model1, args.dualnetwork_target_path)
    target_model1 = torch.load(args.dualnetwork_target_path, map_location=f'cuda:{args.cuda_device}')
    
    # Update target_model2 parameters based on source_model2
    for target_param, source_param in zip(target_model2.parameters(), source_model2.parameters()):
        update_value = source_param.data * args.target_soft_update_ratio 
        target_param.data.copy_(update_value + target_param.data * (1.0 - args.target_soft_update_ratio))
    
    torch.save(target_model2, args.dualnetwork_target2_path)
    target_model2 = torch.load(args.dualnetwork_target2_path, map_location=f'cuda:{args.cuda_device}')


def save_loss_info(all_loss_train_pi, all_loss_q1, all_loss_q2, all_loss_temp, iteration):
    """
    Save the loss information to CSV files.
    Args:
        all_loss_train_pi (list): List containing the loss of the policy network.
        all_loss_q1 (list): List containing the loss of the Q1 network.
        all_loss_q2 (list): List containing the loss of the Q2 network.
        all_loss_temp (list): List containing the temperature loss.
        iteration (int): The current iteration number.
    """

    loss_types = {
        'pi': all_loss_train_pi,
        'q1': all_loss_q1,
        'q2': all_loss_q2,
        'temp': all_loss_temp
    }

    for loss_name, loss_values in loss_types.items():
        file_path = os.path.join(path, f'loss/dual_loss_{iteration}_{loss_name}.csv')
        
        data_frame = pd.DataFrame(loss_values)
        data_frame.to_csv(file_path, index=False, mode='w')



def init_performance(validation_list, iters):
    freeze_support()
    torch.multiprocessing.set_start_method('spawn', force=True)

    # Initialize success information for each episode.
    for subj in validation_list:
        success_idx_file = os.path.join(path, f'terminal/val/val_success_idx/{subj}.txt')
        
        if not os.path.exists(success_idx_file):
            with open(success_idx_file, mode='wt', encoding='utf-8') as f:
                f.write('0')
                
    try:
        with Pool(args.num_process) as pool:
            # Evaluate the best player using multiprocessing.
            pool.starmap(evaluate_best_player_val_p.evaluate_best_player, validation_list)
            pool._cache.clear()  # Ensure all tasks are completed before moving forward.
    finally:
        # Generate a confusion matrix after evaluating all players.
        evaluate_best_player_val_p.confusion(-1, True)

def self_play(validation_list):
    try:
        with Pool(args.num_process) as pool:
            # Using imap to apply self_play_best.self_play to each item in validation_list in parallel.
            pool.imap(self_play_best.self_play, validation_list)
            pool._cache.clear()  # Waiting for all the tasks to complete.
    except Exception as e:
        print(f"An error occurred: {e}")


################ LOAD BUFFER DATA
def replay_buffer_load(replay_buffer):
    load_path = args.backup_path if replay_buffer else args.best_path
    directory = os.path.join(path, 'self_play_backup' if replay_buffer else 'self_play_best_data')
    
    with Pool(10) as pool:
        self_data_append = [pool.apply_async(load_data, (load_path, i)).get() 
                            for i in tqdm(range(len(os.listdir(directory))), desc='Data load')]
    return self_data_append
