import random
import time
import warnings

import numpy as np
import torch
import os
import disconnection
import disconnection_laplace
import util
from Baseline import chev
from DualNetwork import ActorCritic
from General_baseline import chev
from seed import seed_everything

import pyximport; pyximport.install(pyximport=True)
warnings.filterwarnings("ignore") 
import config


args = config.parser.parse_args()

path             = args.path
dualnetwork_best_path   = args.dualnetwork_best_path
DN_OUTPUT_SIZE   = args.action # numer of action
alpha            = 0.9 # weight of reward, value 
DISCOUNT_FACTOR  = args.discount_factor
jeilt            = args.jeilt
True_label       = 0 #MCI = 0, NC = 1
REST             = 116

def play(SUBJ, STOP_POINT, TEMPERATURE, idx):
    ############### DATA
    cross_validation = util.validation_data(args.split)
    cross_validation_label = util.validation_label(args.split)

    data = np.array(cross_validation[SUBJ], dtype=float)
    x0 = torch.Tensor(data).reshape(1, 1, 116, 116)

    ############### ACTOR-CRITIC LOAD
    actor_critic = torch.load(dualnetwork_best_path, map_location='cpu')
    actor_critic.eval()
    
    ############### CLASSIFIER LOAD
    classifier = util.classifier() 
    classifier.eval()

    ############### HISTYRY : TRAIN DATA LIST
    HISTORY = []

    ############### STATE, ADJ ENV INIT
    state = disconnection.State(data, SUBJ, False)
    xx = torch.rand(116,116)
    adj=xx.new_ones(116,116) # adj:116,116
    adj_state=disconnection_laplace.State(adj) # adj_state:116,116

    ############### DATA LABEL
    if cross_validation_label[SUBJ] == [1,0]: True_label = 0
    elif cross_validation_label[SUBJ] == [0,1]: True_label = 1
 
    ############### DATA OUTPUT
    input_data = classifier(x0, adj.float()) # input데이터
    _, input_data_label = torch.max(input_data, 1) # return : max값, max값의 위치

    ############### NUMPY
    terminal_s0 = np.array(state.piece, dtype=float)
    terminal_s0 = torch.tensor(terminal_s0).reshape(1, 1, 116, 116) #terminal_s0:1,116,116

    ############### PI NETWORK
    p0 = util.PI(actor_critic, terminal_s0.float(), adj_state.piece.long(), False)

    ############## TERMINAL 1 (S_0=LABEL)
    if True_label == input_data_label:
        scores = p0.detach().cpu().numpy()[0]

        ############### BOLTZMAN DISTBUTION (TEMPERATURE)
        scores=util.boltzman(scores, TEMPERATURE)

        ############### PLOT
        if args.plt:
            util.pi_p0(scores, SUBJ, 0)

        ############### NEXT ACTION
        action = REST
        print(SUBJ, 'Subject action o : ', action)
        
        ############### HISTORY INIT
        HISTORY.append([state.piece , action, None, None, None, None, None])  # state, action, reward, state_prime, Done, adj, adj_prime

        ############### NEXT STATE
        state = state.piece 

        ############### NEXT ADJ
        adj_state_prime = adj_state.piece.long() 

        ############### EPISODE SUCCES INFO
        with open(path +f'/terminal/val/succes_idx/{SUBJ}_{idx}.txt', mode='rt', encoding='utf-8') as f:
            lines = f.readlines()  
            SUCCES_IDX = float(lines[0])
    
        ############### Reward
        if SUCCES_IDX==1.0: prev_iters=0.5
        elif SUCCES_IDX==0.0: prev_iters=1.0
        elif SUCCES_IDX==-1.0: prev_iters=1.0

        HISTORY[0][2] =  args.reward * prev_iters 
        ############### S PRIME
        HISTORY[0][3] = state 

        ############### DONE
        HISTORY[0][4] = 0

        ############### ADJ
        HISTORY[0][5] = adj_state.piece

        ############### ADJ PRIME
        HISTORY[0][6] = adj_state_prime


        ############### SUCCES INFO ADD
        if SUCCES_IDX==0:
            with open(path + f'/terminal/val/succes_idx/{SUBJ}_{idx}.txt', mode='wt', encoding='utf-8') as f:
                f.writelines(f'{1.0}')

        elif SUCCES_IDX==1.0:
            with open(path + f'/terminal/val/succes_idx/{SUBJ}_{idx}.txt', mode='wt', encoding='utf-8') as f:
                f.writelines(f'{1.0}')

        elif SUCCES_IDX==-1.0:
            with open(path + f'/terminal/val/succes_idx/{SUBJ}_{idx}.txt', mode='wt', encoding='utf-8') as f:
                f.writelines(f'{1.0}')

        with open(path + f'/terminal/val/action/{SUBJ}.txt', mode='at', encoding='utf-8') as f:
            f.writelines('{} \n'.format([116]))
        
        return HISTORY, True

    else:
        adj_state_history, state_history, action_history = [], [], []

        ############### SELFPLAY START
        for i in range(STOP_POINT):
            TEMPERATURE *= 1.0

            ############### DATA NUMPY, VARIABLE
            terminal_s0 = np.array(state.piece)
            terminal_s0 = torch.Tensor(terminal_s0).reshape(1, 1, 116, 116)

            ############### PI NETWORK
            scores = util.PI(actor_critic, terminal_s0, torch.tensor(adj_state.piece), False)
            scores = scores.detach().cpu().numpy()[0]
            scores = util.overlab(action_history, scores)
            
            ############### PLOT
            if args.plt:
                util.pi_p0(scores, SUBJ, i)

            ############### BOLTZMAN DISTBUTION (TEMPERATURE) 
            scores=util.boltzman(scores, TEMPERATURE)

            ############### GET ACTION
            action = np.random.choice(state.legal_actions(), p=scores)
            action_history.append(action)
            HISTORY.append([state.piece , action, None, None, None, None, None])  # state, action, reward, state_prime, done, adj, adj_prime

            ############### NEXT STATE
            state = state.next(action, False)
            state_history.append(state.piece)

            ############### NEXT ADJ
            adj_state = adj_state.next(action)
            adj_state_history.append(adj_state.piece)

            ############### CURRENT STATE OUTPUT
            state_data = classifier(terminal_s0.float(), torch.from_numpy(adj_state.piece).float())  # input데이터 _, x값의 위치
            _, state_data_label = torch.max(state_data, 1) #return : max값, max값의 위치

            ############### TERMINAL 2 (S_N=LABEL)
            if True_label == state_data_label:  
                print(SUBJ, 'Subject action: ', action_history)

                ############### EPISODE SUCCES INFO
                with open(path +f'/terminal/val/succes_idx/{SUBJ}_{idx}.txt', mode='rt', encoding='utf-8') as f:
                    lines = f.readlines()  
                    SUCCES_IDX = float(lines[0])

                ############### HISTORY
                for idx, e in enumerate(range(len(HISTORY))):

                    ############### Reward   
                    episode_len = len(HISTORY)-(e+1)

                    if SUCCES_IDX==1.0: prev_iters=0.5
                    elif SUCCES_IDX==0.0: prev_iters=1.0
                    elif SUCCES_IDX==-1.0: prev_iters=1.0

                    HISTORY[e][2] = args.reward * pow(DISCOUNT_FACTOR, episode_len) * prev_iters 

                    ############### S PRIME
                    if e+1==len(HISTORY): HISTORY[e][3] = state_history[0] 
                    else: HISTORY[e][3] = state_history[e] 

                    ############### DONE
                    if e+1==len(HISTORY): HISTORY[e][4] = 0
                    else: HISTORY[e][4] = 1 

                    ############### ADJ
                    if e==0: HISTORY[e][5] = adj_state.piece
                    else: HISTORY[e][5] = adj_state_history[e]  

                    ############### ADJ PRIME
                    HISTORY[e][6] = adj_state_history[e]

                    ############### SUCCES INFO ADD
                    if SUCCES_IDX==0:
                        with open(path + f'/terminal/val/succes_idx/{SUBJ}_{idx}.txt', mode='wt', encoding='utf-8') as f:
                            f.writelines(f'{1.0}')
                

                    elif SUCCES_IDX==-1.0:
                        with open(path + f'/terminal/val/succes_idx/{SUBJ}_{idx}.txt', mode='wt', encoding='utf-8') as f:
                            f.writelines(f'{-1.0}')

                return HISTORY, False

            ############### TERMINAL 3 (S_N!=LABEL)
            elif i == STOP_POINT-1 or action==116:

                ############### INFO SAVE 
                with open(path +f'/terminal/val/succes_idx/{SUBJ}_{idx}.txt', mode='rt', encoding='utf-8') as f:
                    lines = f.readlines()  
                    SUCCES_IDX = float(lines[0])

                print(SUBJ, 'Subject action: ', action_history)
                
                ############### HISTORY
                for idx, q in enumerate(range(len(HISTORY))):
                    ############### Reward   
                    episode_len = len(HISTORY)-(q+1)

                    if SUCCES_IDX==1.0: prev_iters=-1.0
                    elif SUCCES_IDX==0.0: prev_iters=-1.0
                    elif SUCCES_IDX==-1.0: prev_iters=-0.5

                    HISTORY[q][2] = args.reward * pow(DISCOUNT_FACTOR, episode_len) * prev_iters #* max(state_data[0].tolist())

                    ############### S PRIME
                    if q+1==len(HISTORY): HISTORY[q][3] = state_history[0] #+ util.positional_encoding() 
                    else: HISTORY[q][3] = state_history[q] #+ util.positional_encoding()

                    ############### DONE
                    if q+1==len(HISTORY): HISTORY[q][4] = 0
                    else: HISTORY[q][4] = 1 

                    ############### ADJ
                    if q==0: HISTORY[q][5] = adj_state.piece
                    else: HISTORY[q][5] = adj_state_history[q]  

                    ############### ADJ PRIME
                    HISTORY[q][6] = adj_state_history[q]

                    ############### SUCCES INFO ADD
                    if SUCCES_IDX==0:
                        with open(path + f'/terminal/val/succes_idx/{SUBJ}_{idx}.txt', mode='wt', encoding='utf-8') as f:
                            f.writelines(f'{-1.0}')

                    elif SUCCES_IDX==1:
                        with open(path + f'/terminal/val/succes_idx/{SUBJ}_{idx}.txt', mode='wt', encoding='utf-8') as f:
                            f.writelines(f'{-1.0}')


                return HISTORY, False

############### SELF_PLAY
def self_play(SUBJ):
    start = time.time()
    STOP_POINT, TEMPERATURE, _, _, _,  _, _, _, _ = util.load_hyperparameter()
    t = int( time.time() * 1000.0 )
    time_seed = ( ((t & 0xff000000) >> 24) +
                ((t & 0x00ff0000) >>  8) +
                ((t & 0x0000ff00) <<  8) +
                ((t & 0x000000ff) << 24)   )
    seed_everything(time_seed)
    
    label = util.validation_label(args.split)

    if label[SUBJ] == 1: game_count = args.sp_game_count 
    else: game_count = int(args.sp_game_count * 1.5)

    for idx in range(game_count):
        if not os.path.exists(path + f'/terminal/val/succes_idx/{SUBJ}_{idx}.txt'):
            with open(path + f'/terminal/val/succes_idx/{SUBJ}_{idx}.txt', mode='wt', encoding='utf-8') as f:
                f.writelines(f'{0}')

        h, tf = play(SUBJ, STOP_POINT, TEMPERATURE, idx) 

        util.best_write_data(h, SUBJ, start, idx)
        if tf == True:
            break
        TEMPERATURE*=0.9
    seed_everything(args.seed)





