import os
import warnings
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn
from tqdm import tqdm

import config
import early_stop_dual
import evaluate_best_player_val_p
import loss_idd_dual
import seed
import util
import wandb
from Baseline import chev
from DualNetwork import ActorCritic

import pyximport; pyximport.install() 

os.environ['CUDA_LAUNCH_BLOCKING']  = '1'
os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0,1"
os.environ['MPLCONFIGDIR']          = os.getcwd() + "/configs/"

args                    = config.parser.parse_args()
nbepochs                = args.epochs 
path                    = args.path
best_path               = args.self_data
backup_path             = args.buffer_data
dualnetwork_best_path   = args.dualnetwork_best_path
dualnetwork_best2_path  = args.dualnetwork_best2_path
dualnetwork_target_path = args.dualnetwork_target_path
dualnetwork_target2_path= args.dualnetwork_target2_path
dualnetwork_init_path   = args.dualnetwork_model_init_path
dualnetwork_init2_path  = args.dualnetwork_model_init2_path
cuda_device             = args.cuda_device
use_cuda                = torch.cuda.is_available()

torch.set_num_threads(os.cpu_count()*2)
warnings.filterwarnings("ignore") 
torch.autograd.set_detect_anomaly(False) 
torch.cuda.set_device(args.cuda_device)

############### NN TRAIN START
def train_network(model, history, target_model, model2, target_model2, iters):
    running_loss_pi, running_loss_q1, running_loss_q2 = 0.0, 0.0, 0.0

    ############### BATCH DATA 
    for batch_idx, (state, action, reward, state_prime, done, adj, adj_prime) in enumerate(history):

        ############### TO TENSOR
        state, reward, state_prime, adj, adj_prime = torch.tensor(state, device=torch.device('cuda:0')),\
                                    torch.tensor(reward, device=torch.device('cuda:0')),\
                                    torch.tensor(state_prime, device=torch.device('cuda:0')),\
                                    torch.tensor(adj, device=torch.device('cuda:0')),\
                                    torch.tensor(adj_prime, device=torch.device('cuda:0')),\

        ############### RESHAPE 
        state, adj = state.reshape(-1, 116, 116), adj.reshape(-1, 116, 116)
        state_prime, adj_prime = state_prime.reshape(-1, 116, 116), adj_prime.reshape(-1, 116, 116)

        ############### PUT DATA
        model.put_data([state, action, reward, state_prime, done, adj, adj_prime])

    ############### TRAIN
    loss_pi, loss_q1, loss_q2, loss_temp, _, esv = model.train_net(model, target_model, model2, target_model2, iters) # RETURN : LOSS

    ############### SUMMATION LOSS
    running_loss_pi += loss_pi/(batch_idx+1)
    running_loss_q1 += loss_q1/(batch_idx+1)

    ############### BATCH LOSS
    return loss_pi, loss_q1, loss_temp, _, esv

############### MAIN
def train_main(True_result, iters):
    # initialize the early_stopping object
    early_stopping = early_stop_dual.EarlyStopping(patience=args.patience, verbose=True, delta=0.0)

    allloss_train_pi, allloss_Q1 = [], []
    self_data_append, history, allloss_Q2, allloss_TEMP = [], [], [], []

    ############### SELFPLAY DATA LOAD *REPLAY BUFFER
    if args.replay_buffer:
        with Pool(10) as pool:
            self_data_append = [pool.apply_async(util.load_data, (backup_path, i)).get() for i in tqdm(range(len(os.listdir(path+'/self_play_backup'))), desc='Episode load')]
            pool.close()
            pool.join()
    else:
        with Pool(10) as pool:
            self_data_append = [pool.apply_async(util.load_data, (best_path, i)).get() for i in tqdm(range(len(os.listdir(path+'/self_play_best_data'))), desc='Episode load')]
            pool.close()
            pool.join()

    ############### SAMPLING
    # s, a, r, s_prime, done, adj, adj_prime
    STOP_POINT, _, _, _, _,  _, _, _, _ = util.load_hyperparameter()

    for j in tqdm(range(len(self_data_append)), desc='Data sampling'):
        if len(self_data_append[j])>args.window:

            for z in range(args.sampling_size):
                globals()['history{}'.format(z)] = []

            for history_idx in range(args.sampling_size):
                globals()['history{}'.format(history_idx)].append(self_data_append[j][np.random.randint(len(self_data_append[j])-args.window, len(self_data_append[j]))])
                

            for history_idx2 in range(args.sampling_size):
                history.extend(globals()['history{}'.format(history_idx2)] )

        else:
            for z in range(args.sampling_size//2):
                globals()['history{}'.format(z)] = []

            for history_idx in range(args.sampling_size//2):
                globals()['history{}'.format(history_idx)].append(self_data_append[j][np.random.randint(0, len(self_data_append[j]))])

            for history_idx2 in range(args.sampling_size//2):
                history.extend(globals()['history{}'.format(history_idx2)] )

    ############### MODEL LOAD
    model1 = torch.load(dualnetwork_best_path, map_location=f'cuda:{args.cuda_device}')
    target_model1 = torch.load(dualnetwork_target_path, map_location=f'cuda:{args.cuda_device}')

    model2=torch.load(dualnetwork_best2_path, map_location=f'cuda:{args.cuda_device}')
    target_model2 = torch.load(dualnetwork_target2_path, map_location=f'cuda:{args.cuda_device}')


    ############### TRAIN EPOCHS
    for n_epi in tqdm(range(nbepochs), desc='Train process', leave=False): 


        ############### TRAIN LOSS
        PI, loss_Q1, loss_TEMP, _, esv = train_network(model=model1, history=history, target_model=target_model1, model2=model2, target_model2=target_model2, iters=iters)
        PI, loss_Q1, loss_TEMP = PI.detach().cpu().numpy(), loss_Q1.detach().cpu().numpy(), loss_TEMP.detach().cpu().numpy()
        print('Episode len : ', STOP_POINT, 'PI : ', round(PI.item(),3), 'Q1 : ', round(loss_Q1.item(),3), 'TEMP : ', round(loss_TEMP.item(),3)) #, 'Q2 : ', round(loss_Q2.item(),3)
 
        ############### TARGET NETWORK UPDATE
        if n_epi % args.target_n==0:
            util.target_network(model1, model2, target_model1, target_model2)

        ############### LOSS APPEND
        allloss_train_pi.append(PI)
        allloss_Q1.append(loss_Q1)
        allloss_TEMP.append(loss_TEMP)

        ############### TRAIN LOSS INFO
        util.loss_info(allloss_train_pi, allloss_Q1, allloss_Q2, allloss_TEMP, iters)

        ############### PLOT
        if args.plt:
            loss_idd_dual.loss_idd(iters)

        ############### EARLY STOP_POINT
        if args.val_eval: early_stopping_point = esv
        else: early_stopping_point = -(PI + loss_Q1 + loss_TEMP)

        FINE_TUNE = early_stopping(early_stopping_point, model1, model2, iters) 

        if early_stopping.early_stop:
            print("Early stopping")
            torch.save(model1, path + f'/train_dual_network/arXiv/model_0_{iters}.pt')
            torch.save(model2, path + f'/train_dual_network/arXiv/model_1_{iters}.pt')

            ############### VALIDATION EVALUATE
            validation_data = util.validation_data(args.split)
            validation_list = [(j, iters) for j in range(len(validation_data))]
            
            try:
                pool = Pool(args.num_process)
                pool.starmap(evaluate_best_player_val_p.evaluate_best_player, validation_list)
            finally:
                pool.close()
                pool.join()
                evaluate_best_player_val_p.confusion(iters, True) 
                
            break

    ############### MODEL INFO
    with open(path + '/train_dual_network/model.txt', mode='at',encoding='utf-8') as f:
        f.writelines(f'{model1}')


