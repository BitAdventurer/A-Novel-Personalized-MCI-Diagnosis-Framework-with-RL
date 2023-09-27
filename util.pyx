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

import config
import reward
import seed
from Baseline import chev

args = config.parser.parse_args()
seed.seed_everything(args.seed)  
path=args.path

############### WRITE HYPERPARAMETER
def hyperparameter(STOP_POINT, TEMPERATURE, SP_GAME_COUNT, LR, WD,  MOMENTUM, PROCESS_NUM, BATCH_SIZE, BUFFER_SIZE):
    with open(path + '/hyperparameter.txt', mode='wt', encoding='utf-8') as f:
        f.writelines('{}\n'.format(STOP_POINT))
        f.writelines('{}\n'.format(TEMPERATURE))
        f.writelines('{}\n'.format(SP_GAME_COUNT))
        f.writelines('{}\n'.format(LR))
        f.writelines('{}\n'.format(WD))
        f.writelines('{}\n'.format(MOMENTUM))
        f.writelines('{}\n'.format(PROCESS_NUM))
        f.writelines('{}\n'.format(BATCH_SIZE))
        f.writelines('{}\n'.format(BUFFER_SIZE))

############### LOAD HYPERPARAMETER
def load_hyperparameter():
    with open(path+'/hyperparameter.txt', mode='rt', encoding='utf-8') as f:
        lines = f.readlines()  
        STOP_POINT = int(lines[0])
        TEMPERATURE = float(lines[1])
        SP_GAME_COUNT = int(lines[2])
        LR = float(lines[3])
        WD = float(lines[4])
        MOMENTUM = float(lines[5])
        SOFT_RATTIO = float(lines[6])
        BATCH_SIZE = int(lines[7])
        BUFFER_SIZE = int(lines[8])
    return STOP_POINT, TEMPERATURE, SP_GAME_COUNT, LR, WD,  MOMENTUM, SOFT_RATTIO, BATCH_SIZE, BUFFER_SIZE
    
############### EVALUATE_NETWORK
def soft_update(source, target, tau, tf): # init, Train model
    for target_param, param in zip(target.state_dict().values(), source.state_dict().values()):
        target_param.data.copy_(target_param.data * tau + param.data * (1.0 - tau))
        if tf==True:
            torch.save(target.cuda(), args.dualnetwork_model_init_path)
            torch.save(target.cuda(), args.dualnetwork_best_path)
        else:
            torch.save(target.cuda(), args.dualnetwork_model_init2_path)
            torch.save(target.cuda(), args.dualnetwork_best2_path)

############### EVALUATE_NETWORK
def hard_update(source, target, tf): # init, model
    if tf==True:
        torch.save(target, args.dualnetwork_best_path)
    else:
        torch.save(target, args.dualnetwork_best2_path)

############### TRAIN NETWORK
def load_data(x,y):
    history_path = sorted(Path(x).glob('*.history'))[y]
    with history_path.open(mode='rb') as f:
        return torch.load(f)

############### SELF PLAY
def self_load_data(x, y):
    history_path = sorted(Path(x).glob(y))[0]
    with history_path.open(mode='rb') as f:
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


################ ACTION OVERLAB RULE
def overlab(action_history, p):
    
    if action_history!=[]:
        for i in action_history:
            if i !=116:
                p[i] = 0

    max_p = np.argmax(p)
    scores = np.zeros(len(p))
    scores[max_p] = 1

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


################ SOFTMAX
def softmax(a) :
    a = a.numpy()
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    #y = a / sum(a)
    return y

################ LOAD CRITIC
def Q(model, s, a, m, tm):
    if tm==True: 
        model.train()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

    else: model.eval()

    return model.Q(s.float(), a.float(), m, tm).squeeze()

################ LOAD ACTOR
def PI(model, s, m, tm):
    if tm==True: 
        model.train()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else: model.eval()
    return model.pi(s, m, tm, softmax_dim=1)

################ LOAD TEMPERATURE
def T(model, s, adj, tm):
    if tm==True: 
        model.train()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else: model.eval()
    return model.temp(s.float(), adj.float()).squeeze()

################ LOAD CLASSIFIER
def classifier():
    return torch.load(path+f'/baseline/fold{args.fold}/checkpoint/{args.split}checkpoint.pt', map_location='cpu')

################ TARGET1, TARGET2 NETWORK UPDATE
def target_network(model1, model2, target_model1, target_model2):
    print('-----Target Network Change-----')
    for target_param, param in zip(target_model1.parameters(), model1.parameters()):
        target_param.data.copy_(param.data * args.target_soft_update_ratio + target_param.data * (1.0 - args.target_soft_update_ratio))

    torch.save(target_model1, args.dualnetwork_target_path)
    target_model1 = torch.load(args.dualnetwork_target_path, map_location=f'cuda:{args.cuda_device}')

    for target_param, param in zip(target_model2.parameters(), model2.parameters()):
        target_param.data.copy_(param.data * args.target_soft_update_ratio + target_param.data * (1.0 - args.target_soft_update_ratio))

    torch.save(target_model2, args.dualnetwork_target2_path)
    target_model2 = torch.load(args.dualnetwork_target2_path, map_location=f'cuda:{args.cuda_device}')

################ NETWORK LOSS SAVE
def loss_info(allloss_train_pi, allloss_Q1, allloss_Q2, allloss_TEMP, iters):
    dataframe_loss_pi = pd.DataFrame(allloss_train_pi)
    data = pd.concat([dataframe_loss_pi], axis=1)
    data.to_csv(path + f'/loss/dual_loss_{iters}_pi.csv', index = False, mode ='w')
    dataframe_loss_v = pd.DataFrame(allloss_Q1)
    data = pd.concat([dataframe_loss_v], axis=1)
    data.to_csv(path + f'/loss/dual_loss_{iters}_q1.csv', index = False, mode ='w')
    dataframe_loss = pd.DataFrame(allloss_Q2)
    data = pd.concat([dataframe_loss], axis=1)
    data.to_csv(path + f'/loss/dual_loss_{iters}_q2.csv', index = False, mode ='w')
    dataframe_loss = pd.DataFrame(allloss_TEMP)
    data = pd.concat([dataframe_loss], axis=1)
    data.to_csv(path + f'/loss/dual_loss_{iters}_temp.csv', index = False, mode ='w')

################ INIT ITERATION PERFORMANCE
def init_performance(validation_list, iters):
    from multiprocessing import Pool, freeze_support

    import evaluate_best_player_val_p
    freeze_support()
    torch.multiprocessing.set_start_method('spawn', force=True)

    for subj in range(len(validation_list)):

    ############### EPISODE SUCCES INFO

    for subj in range(len(validation_list)):
        if not os.path.exists(path + f'/terminal/val/val_succes_idx/{subj}.txt'):
            with open(path + f'/terminal/val/val_succes_idx/{subj}.txt', mode='wt', encoding='utf-8') as f:
                f.writelines(f'{0}')

    try:
        pool = Pool(args.num_process)
        pool.starmap(evaluate_best_player_val_p.evaluate_best_player, validation_list)
    finally:
        pool.close()
        pool.join()
        evaluate_best_player_val_p.confusion(-1, True) 



################ PLAY SELFPLAY
def self_play(validation_list):
    import self_play_best

    ############### SELF-PLAY
    try:
        pool = Pool(args.num_process)
        pool.imap(self_play_best.self_play,validation_list)
    finally:
        pool.close()
        pool.join()

################ LOAD BUFFER DATA
def replay_buffer_load(replay_buffer):
    if replay_buffer==True:
        with Pool(10) as pool:
            self_data_append = [pool.apply_async(load_data, (args.backup_path, i)).get() for i in tqdm(range(len(os.listdir(path+'/self_play_backup'))), desc='Data load')]
            pool.close()
            pool.join()
    else:
        with Pool(10) as pool:
            self_data_append = [pool.apply_async(load_data, (args.best_path, i)).get() for i in tqdm(range(len(os.listdir(path+'/self_play_best_data'))), desc='Data load')]
            pool.close()
            pool.join()
        return self_data_append

