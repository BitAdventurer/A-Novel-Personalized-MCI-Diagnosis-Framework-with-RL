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
# initialize the early_stopping object
early_stopping = early_stop_dual.EarlyStopping(patience=args.patience, verbose=True, delta=0.0)

allloss_train_pi, allloss_Q1 = [], []
self_data_append, history, allloss_Q2, allloss_TEMP = [], [], [], []
STOP_POINT, *_ = util.load_hyperparameter()

def train_network(model, history, target_model, model2, target_model2, iters):
    """
    Trains the provided model using data from history.
    
    :param model: The main model to be trained.
    :param history: The data used for training the model.
    :param target_model: The target model used during training.
    :param model2: An additional model involved in training.
    :param target_model2: The target of the additional model involved in training.
    :param iters: The number of iterations to train for.
    :return: Tuple containing the losses and other return values from the training process.
    """
    # Constants
    TENSOR_DIM = 116
    CUDA_DEVICE = 'cuda:0'
    
    # Initialize running losses
    loss_pi, loss_q1, loss_q2 = 0.0, 0.0, 0.0
    
    # Process Batch Data
    for batch_idx, (state, action, reward, state_prime, done, adj, adj_prime) in enumerate(history):
        
        # Convert to tensor and reshape
        state, reward, state_prime, adj, adj_prime = map(
            lambda x: torch.tensor(x, device=torch.device(CUDA_DEVICE)).reshape(-1, TENSOR_DIM, TENSOR_DIM),
            [state, reward, state_prime, adj, adj_prime]
        )
        
        # Put data into model
        model.put_data([state, action, reward, state_prime, done, adj, adj_prime])
    
    # Perform Training and Compute Loss
    loss_pi, loss_q1, _, loss_temp, _, esv = model.train_net(
        model, target_model, model2, target_model2, iters
    )
    
    # Return Losses and Other Values
    return loss_pi, loss_q1, loss_temp, _, esv


############### MAIN
def train_main(True_result, iters):
    """
    Main function to handle the training process.
    
    :param true_result: None
    :param iters: Number of iterations
    """

    def load_data(path):
        with Pool(10) as pool:
            data = [pool.apply_async(util.load_data, (path, i)).get() for i in tqdm(range(len(os.listdir(path))), desc='Episode load')]
        return data

    # Load Data
    if args.replay_buffer:
        self_data_append = load_data(path + '/self_play_backup')
    else:
        self_data_append = load_data(path + '/self_play_best_data')
        
    ############### SAMPLING
    history = []  # It's assumed that history is a list. Modify as needed if it's another data structure.
    
    def sample_and_extend(data, sample_size):
        sampled_histories = [[] for _ in range(sample_size)]
        for idx in range(sample_size):
            sampled_histories[idx].append(data[np.random.randint(len(data))])
        return [item for sublist in sampled_histories for item in sublist]  # Flatten and return

    for data in tqdm(self_data_append, desc='Data sampling'):
        sample_size = args.sampling_size if len(data) > args.window else args.sampling_size // 2
        window_start = max(0, len(data) - args.window)  # This ensures that the sampling window doesn't go negative
        sampled_data = data[window_start:]
        history.extend(sample_and_extend(sampled_data, sample_size))

    def load_model(path, cuda_device):
        try:
            model = torch.load(path, map_location=f'cuda:{cuda_device}')
            return model
        except Exception as e:
            print(f"Error loading the model from path {path}: {str(e)}")
            return None
            
    model1 = load_model(dualnetwork_best_path, args.cuda_device)
    target_model1 = load_model(dualnetwork_target_path, args.cuda_device)
    model2 = load_model(dualnetwork_best2_path, args.cuda_device)
    target_model2 = load_model(dualnetwork_target2_path, args.cuda_device)


def training_process(nbepochs, model1, history, target_model1, model2, target_model2, iters, args):
    for epoch in tqdm(range(nbepochs), desc='Train process', leave=False): 
        # Calculate loss through training
        PI, loss_Q1, loss_TEMP, _, esv = train_network(
            model=model1, 
            history=history, 
            target_model=target_model1, 
            model2=model2, 
            target_model2=target_model2, 
            iters=iters
        )
        
        # Transfer data from GPU to CPU and convert to numpy array
        PI, loss_Q1, loss_TEMP = PI.detach().cpu().numpy(), loss_Q1.detach().cpu().numpy(), loss_TEMP.detach().cpu().numpy()
        
        # Print training information for the current epoch
        print('Episode len : ', STOP_POINT, 'PI : ', round(PI.item(), 3), 'Q1 : ', round(loss_Q1.item(), 3), 'TEMP : ', round(loss_TEMP.item(), 3))
        
        # Update target network at specified epochs
        if epoch % args.target_n == 0:
            util.target_network(model1, model2, target_model1, target_model2)

        # Store loss values
        allloss_train_pi.append(PI)
        allloss_Q1.append(loss_Q1)
        allloss_TEMP.append(loss_TEMP)

        # Print training loss information
        util.loss_info(allloss_train_pi, allloss_Q1, allloss_Q2, allloss_TEMP, iters)

        # Generate plot
        if args.plt:
            loss_idd_dual.loss_idd(iters)

        # Determine whether to perform early stopping
        early_stopping_point = esv if args.val_eval else -(PI + loss_Q1 + loss_TEMP)
        
        # Execute early stopping logic
        FINE_TUNE = early_stopping(early_stopping_point, model1, model2, iters) 
        if early_stopping.early_stop:
            print("Early stopping")
            
            # Save the models
            torch.save(model1, path + f'/train_dual_network/arXiv/model_0_{iters}.pt')
            torch.save(model2, path + f'/train_dual_network/arXiv/model_1_{iters}.pt')

            # Evaluate model with validation data
            evaluate_best_player_val_p(iters)
            break
    
    # Write model information
    with open(path + '/train_dual_network/model.txt', mode='at', encoding='utf-8') as f:
        f.writelines(f'{model1}')

