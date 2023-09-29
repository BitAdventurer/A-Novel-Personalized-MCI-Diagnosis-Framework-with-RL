import copy
import math
import os
import random
import madgrad
import numpy as np
import torch
import torch.nn as nn
import torch_optimizer as optim
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader  # 데이터로더
from torch.utils.data import TensorDataset  # 텐서데이터셋
import wandb
import config
import gcn_util
import util

import pyximport; pyximport.install()
torch.autograd.set_detect_anomaly(True)

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
args                    = config.parser.parse_args()
gamma                   = args.discount_factor
batch_size              = args.batch_size
use_cuda                = torch.cuda.is_available()
cuda_device             = args.cuda_device
dualnetwork_best_path   = args.dualnetwork_best_path
dualnetwork_best2_path  = args.dualnetwork_best2_path

# Initialize Policy weights
def weights_init_xavier_normal_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0)

def weights_init_kaiming_uniform_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)

class ActorCritic(nn.Module):
    def __init__(self, in_features, out_features, K, bias=True, weight_init='xavier'):
        super(ActorCritic, self).__init__()
        self.data = []
        
        ############### ACTOR
        # Layer Normalization for the Actor's input
        self.bnpi = torch.nn.LayerNorm(13456)
        
        # Defining Actor's state and hidden layers with Layer Normalization
        self.state_pi = nn.Linear(in_features=13456, out_features=args.state_emb_layer, bias=bias)
        self.state_pi_bn = torch.nn.LayerNorm(args.state_emb_layer)
        
        self.fcp_1 = nn.Linear(in_features=args.state_emb_layer, out_features=args.actor_layer1, bias=bias)
        self.fcp_bn1 = torch.nn.LayerNorm(args.actor_layer1)
        
        self.fcp_2 = nn.Linear(in_features=args.actor_layer1, out_features=args.actor_layer2, bias=bias)
        self.fcp_bn2 = torch.nn.LayerNorm(args.actor_layer2)
        
        self.fcp = nn.Linear(in_features=args.actor_layer2, out_features=args.actor_layer, bias=bias)
        self.fcp_bn = torch.nn.LayerNorm(args.actor_layer)
    
        ############### CRITIC
        # Layer Normalization for the Critic's input
        self.bnq = torch.nn.LayerNorm(13456)
        
        self.action = nn.Linear(in_features=1, out_features=args.action_emb_layer, bias=bias)        
        self.action_bn = torch.nn.LayerNorm(args.action_emb_layer)
        
        # Layer Normalization for the concatenated state and action layers
        self.bn = torch.nn.LayerNorm(args.state_emb_layer + args.action_emb_layer)
        
        # Defining Critic's state and hidden layers with Layer Normalization
        self.state_q = nn.Linear(in_features=13456, out_features=args.state_emb_layer, bias=bias)
        self.state_q_bn = torch.nn.LayerNorm(args.state_emb_layer)
        
        self.fcq_1 = nn.Linear(in_features=args.state_emb_layer + args.action_emb_layer, out_features=args.critic_layer1, bias=bias)
        self.fcq_bn1 = torch.nn.LayerNorm(args.critic_layer1)
        
        self.fcq_2 = nn.Linear(in_features=args.critic_layer1, out_features=args.critic_layer2, bias=bias)
        self.fcq_bn2 = torch.nn.LayerNorm(args.critic_layer2)
        
        self.fcq = nn.Linear(in_features=args.critic_layer2, out_features=args.critic_layer, bias=bias)
        self.fcq_bn = torch.nn.LayerNorm(args.critic_layer)
    
        ############### TEMP
        # Defining Temperature state and hidden layers with Layer Normalization
        self.state_temp = nn.Linear(in_features=13456, out_features=args.state_emb_layer, bias=bias)
        self.state_temp_bn = torch.nn.LayerNorm(args.state_emb_layer)
        
        self.fctemp_1 = nn.Linear(in_features=args.state_emb_layer, out_features=args.temp_layer1, bias=bias)
        self.fctemp_bn1 = torch.nn.LayerNorm(args.temp_layer1)
        
        self.fctemp_2 = nn.Linear(in_features=args.temp_layer1, out_features=args.temp_layer2, bias=bias)
        self.fctemp_bn2 = torch.nn.LayerNorm(args.temp_layer2)
        
        self.fctemp = nn.Linear(in_features=args.temp_layer2, out_features=args.temp_layer, bias=bias)
        self.fctemp_bn = torch.nn.LayerNorm(args.temp_layer)
    
        # Applying the specified weight initialization
        if weight_init == "xavier":
            self.apply(weights_init_xavier_normal_)
        else:
            self.apply(weights_init_kaiming_uniform_)
    def Q(self, s, a, adj, tm):
        # Setting weight depending on the tm parameter and ensuring GPU compatibility
        we = torch.tensor(0.5).to('cuda') if tm else torch.tensor(0.5)
        
        # Normalizing and reshaping the state
        s = F.normalize(s, dim=1, p=2).reshape(s.size(0), -1)
        
        # Passing the state through normalization layer and activation function
        s = F.prelu(self.bnq(s), we)
        
        # Further processing of state through layers
        s = F.prelu(self.state_q_bn(self.state_q(s)), we)
        
        # One-hot encoding of action if specified
        a = F.one_hot(a.to(torch.int64), num_classes=117) if args.one_hot else a
        a = self.action(torch.tensor(a, dtype=torch.float))
        
        # Further processing of the action-state combination through layers
        out = torch.concat((s.reshape(s.size(0), args.state_emb_layer), 
                            a.reshape(a.size(0), args.action_emb_layer)), dim=1)
        out = F.prelu(self.bn(out), we)
        
        # Passing through Critic layers to get Q-value
        q = F.prelu(self.fcq_bn1(self.fcq_1(out)), we)
        q = F.prelu(self.fcq_bn2(self.fcq_2(q)), we)
        
        # If feature flag is on, saving the Critic feature
        if args.feature:
            with open(args.path + f'/value/critic_feature.txt', mode='at', encoding='utf-8') as f:
                f.writelines(f'{q.tolist()[0]}\n')
        
        # Returning the soft sign of final Critic output
        return F.softsign(self.fcq(q))

    ### Pi Function
    def pi(self, s, adj, tm, softmax_dim=1):
        # Setting weight depending on tm parameter and ensuring GPU compatibility
        we = torch.tensor(0.5).to('cuda') if tm else torch.tensor(0.5)
        
        # Normalizing, reshaping, and processing the state through Actor layers to get policy
        p = F.prelu(self.bnpi(F.normalize(s, dim=1, p=2).reshape(p.size(0), -1)), we)
        p = F.prelu(self.state_pi_bn(self.state_pi(p)), we)
        p = F.prelu(self.fcp_bn1(self.fcp_1(p)), we)
        p = F.prelu(self.fcp_bn2(self.fcp_2(p)), we)
        
        # Returning the softmax of final Actor output
        return F.softmax(self.fcp(p), dim=softmax_dim)
        
    ### Temp Function
    def temp(self, s, adj):
        # Normalizing, reshaping, and processing the state through Temp layers to get temperature
        t = F.normalize(s, dim=1, p=2).reshape(t.size(0), -1)
        t = F.relu(self.state_temp_bn(self.state_temp(t)))
        t = F.relu(self.fctemp_bn1(self.fctemp_1(t)))
        t = F.relu(self.fctemp_bn2(self.fctemp_2(t)))
        
        # Returning the relu of final Temp output
        return F.relu(self.fctemp(t))


    def put_data(self, transition):
        # Receiving and appending each transition to the data list
        self.data.append(transition)
    
    def make_batch(self):
        # Initializing lists to store different elements of each transition
        s_lst, a_lst, r_lst, s_prime_lst, done_lst, adj_lst, adj_prime_lst = [], [], [], [], [], [], []
        for transition in self.data:
            # Extracting and appending elements from each transition to the respective lists
            s, a, r, s_prime, done, adj, adj_prime = transition
            s_lst.append(s.clone().cpu().numpy().reshape(-1, 116, 116))  # to numpy array
            a_lst.append([a])  # to list
            r_lst.extend([r])  # extending the list with reward value
            s_prime_lst.append(s_prime.clone().cpu().numpy().reshape(-1, 116, 116))
            done_mask = 0.0 if done else 1.0  # Creating a mask for done flag
            done_lst.append(done_mask)  # appending the mask to the list
            adj_lst.append(adj.clone().cpu().numpy().reshape(-1, 116, 116))
            adj_prime_lst.append(adj_prime.clone().cpu().numpy().reshape(-1, 116, 116))

        # Converting lists to tensors, setting to the required gradient and device (GPU)
        s_batch, a_batch, r_batch, s_prime_batch, done_batch, adj_batch, adj_prime_batch \
            = torch.tensor(s_lst, dtype=torch.float, requires_grad=True, device=torch.device('cuda:0')),\
              torch.tensor(a_lst, dtype=torch.float, requires_grad=True, device=torch.device('cuda:0')), \
              torch.tensor(r_lst, dtype=torch.float, requires_grad=True, device=torch.device('cuda:0')),\
              torch.tensor(s_prime_lst, dtype=torch.float, requires_grad=True, device=torch.device('cuda:0')), \
              torch.tensor(done_lst, dtype=torch.float, requires_grad=True, device=torch.device('cuda:0')),\
              torch.tensor(adj_lst, dtype=torch.float, requires_grad=True, device=torch.device('cuda:0')),\
              torch.tensor(adj_prime_lst, dtype=torch.float, requires_grad=True, device=torch.device('cuda:0'))

        # Emptying the data list for the next batch
        self.data = []

        # Returning batches of tensors
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch, adj_batch, adj_prime_batch


     def train_net(self, model, target_model, model2, target_model2, iters):
        
        # Setting up Gradient Scaler for Mixed Precision Training
        scaler = torch.cuda.amp.GradScaler()
        
        # Loading Hyperparameters: Learning Rate, Weight Decay, etc.
        _, _, _, lr, wd, _, _, _, _ = util.load_hyperparameter()
    
        # Setting up the Optimizers based on user choice.
        # The optimizer is used to update the weights of the model to minimize the loss.
        self.configure_optimizers(args, model, model2, lr, wd)
    
        # Setting up Learning Rate Schedulers to adjust the learning rate during training.
        scheduler, scheduler2 = self.configure_schedulers()
    
        # Making Data Batches for training
        s, a, r, s_prime, done, adj, adj_prime = self.make_batch()
        
        # Loading Data into PyTorch DataLoader for efficient data loading during training.
        s_loader = self.configure_dataloader(s, a, r, s_prime, done, adj, adj_prime)
        
        # Iterating over each batch of data and performing training operations.
        for s_batch, a_batch, r_batch, s_prime_batch, done_batch, adj_batch, adj_prime_batch in s_loader:
            # Perform necessary operations and computations for training the model
            self.training_step(model, target_model, model2, target_model2, s_batch, a_batch, r_batch, s_prime_batch, done_batch, adj_batch, adj_prime_batch, scaler, scheduler, scheduler2)
    
        # Logging training metrics to W&B (Weights & Biases) for visualization and analysis.
        wandb.log({'Soft Policy': soft_policy_error, 'Soft Value': soft_q1_value_error, 'Temperature': temperature_error})
    
        return soft_policy_error, soft_q1_value_error, 0, temperature_error, 0, val_es
    
    def configure_optimizers(self, args, model, model2, lr, wd):
        # Configuring Optimizer based on user input
        # Refer to PyTorch Documentation for more details on each optimizer: https://pytorch.org/docs/stable/optim.html
        # args.optimizer is a string denoting the chosen optimizer.
        optimizer_dict = {
            'RAdam': (optim.RAdam, optim.RAdam),
            'adam': (torch.optim.Adam, torch.optim.Adam),
            'sgd': (torch.optim.SGD, torch.optim.SGD),
            'rmsprop': (torch.optim.RMSprop, torch.optim.RMSprop),
            'madgrad': (madgrad.MADGRAD, madgrad.MADGRAD),
            'adamw': (torch.optim.AdamW, torch.optim.AdamW)
        }
        self.optimizer, self.optimizer2 = [opt(filter(lambda p: p.requires_grad, model_param.parameters()), lr=lr, weight_decay=wd) for opt, model_param in zip(optimizer_dict.get(args.optimizer, (torch.optim.Adam, torch.optim.Adam)), (model, model2))]
        
    def configure_schedulers(self):
        # Setting up Learning Rate Schedulers to adjust the learning rate during training.
        scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        scheduler2 = lr_scheduler.ExponentialLR(self.optimizer2, gamma=0.99)
        return scheduler, scheduler2
        
    def configure_dataloader(self, s, a, r, s_prime, done, adj, adj_prime):
        # Configuring the DataLoader for the batches created.
        s_dataset = TensorDataset(s, a, r, s_prime, done, adj, adj_prime)
        s_loader = DataLoader(s_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, drop_last=True)
        return s_loader
        
    def training_step(self, model, target_model, model2, target_model2, s_batch, a_batch, r_batch, s_prime_batch, done_batch, adj_batch, adj_prime_batch, scaler, scheduler, scheduler2):
        """
        Performs a single step of training with the given batch of data.
        
        Args:
            model, target_model, model2, target_model2: PyTorch models involved in training.
            s_batch, a_batch, r_batch, s_prime_batch, done_batch, adj_batch, adj_prime_batch: Batches of data.
            scaler: Gradient Scaler for mixed precision training.
            scheduler, scheduler2: Learning rate schedulers.
        """
        
        # Prepare the data and models for training
        self.prepare_for_training(s_batch, a_batch, adj_batch, adj_prime_batch)
        
        # Zero the parameter gradients
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        
        # Generate actions and compute respective scores
        action_s, action_prime, PI, PI_PRIME, TEMP, Q1, target_Q1, target_Q2 = self.generate_actions_and_scores(s_batch, s_prime_batch, adj_batch, adj_prime_batch, model, target_model, target_model2, action_s)
        
        # Compute Losses and Objective Function
        objective_function, soft_policy_error, soft_q1_value_error, temperature_error = self.compute_losses_and_objective(PI, Q1, r_batch, done_batch, target_Q, TEMP, action_prime, PI_PRIME)
        
        # Backward and optimize
        self.backward_and_optimize(scaler, soft_q1_value_error, soft_policy_error, temperature_error)
        
        # Update Schedulers and save the models
        self.update_schedulers_and_save_models(scheduler, scheduler2, model, model2)
        
        return soft_policy_error, soft_q1_value_error, 0, temperature_error, 0, val_es
    
    def prepare_for_training(self, s_batch, a_batch, adj_batch, adj_prime_batch):
        """
        Transfers data to the appropriate device.
        """
        # Assuming use_cuda is a class variable, indicating whether to use CUDA or not
        if self.use_cuda:
            s_batch, a_batch, adj_batch, adj_prime_batch = s_batch.cuda(), a_batch.cuda(), adj_batch.cuda(), adj_prime_batch.cuda()
        
        return s_batch, a_batch, adj_batch, adj_prime_batch

    def generate_actions_and_scores(self, s_batch, s_prime_batch, adj_batch, adj_prime_batch, model, target_model, target_model2):
        """
        Generates actions and computes their respective scores.
        """
        action_s = torch.tensor([[x.item()] for x in a_batch], device=torch.device('cuda:0'), dtype=int)
        action_prime = []
        for p in util.PI(model, s_prime_batch, adj_prime_batch, True):
            scores = util.softmax(p.clone().detach().cpu())
            action_prime.append([np.random.choice(np.arange(self.args.action), p=scores)])
        action_prime = torch.tensor(action_prime, device=torch.device('cuda:0'), dtype=int)
        
        PI, PI_PRIME, TEMP, Q1, target_Q1, target_Q2 = self.compute_scores(model, target_model, target_model2, s_batch, s_prime_batch, adj_batch, adj_prime_batch, action_s, action_prime)
        
        return action_s, action_prime, PI, PI_PRIME, TEMP, Q1, target_Q1, target_Q2
    
    def compute_scores(self, model, target_model, target_model2, s_batch, s_prime_batch, adj_batch, adj_prime_batch, action_s, action_prime):
        """
        Computes various scores required for loss computation.
        """
        PI = util.PI(model, s_batch, adj_batch, True)
        PI = torch.gather(PI, 1, action_s)
        
        PI_PRIME = util.PI(model, s_prime_batch, adj_prime_batch, True)
        PI_PRIME = torch.gather(PI_PRIME, 1, action_prime)
        
        TEMP = util.T(model, s_batch, adj_batch, True)
        Q1 = util.Q(model, s_batch, action_s, adj_batch, True)
        
        target_Q1 = util.Q(target_model, s_prime_batch, action_prime, adj_prime_batch, True)
        target_Q2 = util.Q(target_model2, s_prime_batch, action_prime, adj_prime_batch, True)
        
        return PI, PI_PRIME, TEMP, Q1, target_Q1, target_Q2
    
    def compute_losses_and_objective(self, PI, Q1, r_batch, done_batch, target_Q1, target_Q2, TEMP, action_prime, PI_PRIME):
        """
        Computes the losses and the objective function.
        """
        target_Q = torch.minimum(target_Q1, target_Q2)
        TD_target = r_batch + self.gamma * (target_Q + TEMP.detach() * torch.log10(PI_PRIME.detach())) * done_batch
        soft_q1_value_error = F.smooth_l1_loss(Q1, TD_target)
        soft_policy_error = TEMP.detach() * torch.log10(PI) - Q1.detach()
        temperature_obj = -TEMP * (torch.log10(PI.detach()) + torch.tensor(self.args.target_entropy))
        
        objective_function = self.args.policy * soft_policy_error.mean() + self.args.value * soft_q1_value_error.mean() + self.args.temp * temperature_obj.mean()
        
        return objective_function, soft_policy_error, soft_q1_value_error, temperature_obj
    
    def backward_and_optimize(self, scaler, soft_q1_value_error, soft_policy_error, temperature_error):
        """
        Performs backpropagation and optimization.
        """
        scaler.scale(soft_q1_value_error).backward(retain_graph=True)
        scaler.scale(soft_policy_error).backward()
        scaler.scale(temperature_error).backward()
        
        if self.args.w_clamp:
            self.clamp_weights(self.model)
            self.clamp_weights(self.model2)
        
        if self.args.g_clamp:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.g_clamp_v)
            torch.nn.utils.clip_grad_norm_(self.model2.parameters(), self.args.g_clamp_v)
        
        scaler.step(self.optimizer)
        scaler.update()
        
        scaler.step(self.optimizer2)
        scaler.update()
    
    def update_schedulers_and_save_models(self, scheduler, scheduler2, model, model2):
        """
        Updates learning rate schedulers and saves the models.
        """
        scheduler.step()
        scheduler2.step()
        torch.save(model, self.args.dualnetwork_model_init_path)
        torch.save(model2, self.args.dualnetwork_model_init2_path)

if __name__ == "__main__":
    # Set the start method for multiprocessing to 'spawn'.
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    # Import and set a random seed for reproducibility.
    import seed
    seed.seed_everything(args.seed)
    
    # Call freeze_support to allow the script to be frozen (converted into a standalone executable).
    from multiprocessing import freeze_support
    freeze_support()
    
    # Initialize two ActorCritic models with different weight initialization methods.
    net_x = ActorCritic(in_features=args.in_feature, out_features=args.out_feature, K=args.k, weight_init='xavier')
    net_k = ActorCritic(in_features=args.in_feature, out_features=args.out_feature, K=args.k, weight_init='kaiming')
    
    # Write the current working directory to a file named 'path.txt'.
    with open('path.txt', mode='wt', encoding='utf-8') as f:
        f.writelines('{}'.format(os.getcwd()))
    
    # Define the path where the models will be saved.
    path = args.path
    
    # Save models to specific paths.
    torch.save(net_x, os.path.join(path, 'train_dual_network', 'best.pt'))
    torch.save(net_k, os.path.join(path, 'train_dual_network', 'best2.pt'))
    torch.save(net_x, os.path.join(path, 'train_dual_network', 'target.pt'))
    torch.save(net_k, os.path.join(path, 'train_dual_network', 'target2.pt'))
    torch.save(net_x, os.path.join(path, 'train_dual_network', 'arXiv', 'origin.pt'))
    torch.save(net_k, os.path.join(path, 'train_dual_network', 'arXiv', 'origin2.pt'))

