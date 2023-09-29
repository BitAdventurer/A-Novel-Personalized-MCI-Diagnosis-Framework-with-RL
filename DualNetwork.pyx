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
    def __init__(self, in_features, out_features, K,bias=True, weight_init='xavier'):
        super(ActorCritic, self).__init__()
        self.data = []

        ############### ACTOR
        self.bnpi = torch.nn.LayerNorm(13456)

        self.state_pi = (nn.Linear(in_features = 13456, out_features = args.state_emb_layer, bias=bias))
        self.state_pi_bn = torch.nn.LayerNorm(args.state_emb_layer)

        self.fcp_1 = nn.Linear(in_features = args.state_emb_layer, out_features = args.actor_layer1, bias=bias) # 4096
        self.fcp_bn1 = torch.nn.LayerNorm(args.actor_layer1)

        self.fcp_2 = nn.Linear(in_features = args.actor_layer1, out_features = args.actor_layer2, bias=bias)
        self.fcp_bn2 = torch.nn.LayerNorm(args.actor_layer2)

        self.fcp = nn.Linear(in_features = args.actor_layer2, out_features=args.actor_layer, bias=bias)
        self.fcp_bn = torch.nn.LayerNorm(args.actor_layer)

        ############### CRITIC
        self.bnq = torch.nn.LayerNorm(13456)
        if args.one_hot:
            self.action = (nn.Linear(in_features = 117, out_features = args.action_emb_layer, bias=bias))
        else:
            self.action = (nn.Linear(in_features = 1, out_features = args.action_emb_layer, bias=bias))

        self.action_bn = torch.nn.LayerNorm(args.action_emb_layer)
        #torch.clamp(self.action.weight.detach(), min=-10*(torch.mean(self.action.weight.detach()) + torch.std(self.action.weight.detach())), max=10*(torch.mean(self.action.weight.detach()) + torch.std(self.action.weight.detach())))

        self.bn = torch.nn.LayerNorm(args.state_emb_layer + args.action_emb_layer)

        self.state_q = (nn.Linear(in_features = 13456, out_features = args.state_emb_layer, bias=bias))
        self.state_q_bn = torch.nn.LayerNorm(args.state_emb_layer)

        self.fcq_1 = (nn.Linear(in_features = args.state_emb_layer + args.action_emb_layer, out_features = args.critic_layer1, bias=bias))
        self.fcq_bn1 = torch.nn.LayerNorm(args.critic_layer1)

        self.fcq_2 = (nn.Linear(in_features = args.critic_layer1, out_features= args.critic_layer2, bias=bias))
        self.fcq_bn2 = torch.nn.LayerNorm(args.critic_layer2)

        self.fcq = (nn.Linear(in_features = args.critic_layer2, out_features=args.critic_layer, bias=bias))
        self.fcq_bn = torch.nn.LayerNorm(args.critic_layer)


        ############### TEMP
        self.state_temp = (nn.Linear(in_features = 13456, out_features = args.state_emb_layer, bias=bias))
        self.state_temp_bn = torch.nn.LayerNorm(args.state_emb_layer)

        self.fctemp_1 = (nn.Linear(in_features = args.state_emb_layer, out_features = args.temp_layer1, bias=bias))
        self.fctemp_bn1 = torch.nn.LayerNorm(args.temp_layer1)

        self.fctemp_2 = (nn.Linear(in_features = args.temp_layer1, out_features= args.temp_layer2, bias=bias))
        self.fctemp_bn2 = torch.nn.LayerNorm(args.temp_layer2)

        self.fctemp = (nn.Linear(in_features = args.temp_layer2, out_features=args.temp_layer, bias=bias))
        self.fctemp_bn = torch.nn.LayerNorm(args.temp_layer)

        if weight_init=="xavier":
            self.apply(weights_init_xavier_normal_)
        else:
            self.apply(weights_init_kaiming_uniform_)

    def Q(self, s, a, adj, tm):

        if tm==True: we=torch.tensor(0.5).to('cuda')
        else: we=torch.tensor(0.5)

        s = F.normalize(s, dim=1, p=2)
        
        s = s.reshape(s.size(0),-1)

        s = self.bnq(s)
        s = F.prelu(s, we)

        s = self.state_q(s)

        s = self.state_q_bn(s)
        s = F.prelu(s, we)

        if args.one_hot: a = F.one_hot(a.to(torch.int64), num_classes=117)
        else: a = a 

        a = self.action(torch.tensor(a, dtype=torch.float))

        a = self.action_bn(a)
        s = F.prelu(s, we)

        out = torch.concat((s.reshape(s.size(0),args.state_emb_layer), a.reshape(a.size(0),args.action_emb_layer)), dim=1)

        out = self.bn(out)
        out = F.prelu(out, we)


        q = self.fcq_1(out)
        q = self.fcq_bn1(q)
        q = F.prelu(q, we)

        q = self.fcq_2(q)
        q = self.fcq_bn2(q)
        q = F.prelu(q, we)

        ############### Critic feature
        if args.feature:
            with open(args.path + f'/value/critic_feature.txt', mode='at', encoding='utf-8') as f:
                f.writelines(f'{q.tolist()[0]}\n')

        q = self.fcq(q)
        return F.softsign(q)

    def pi(self, s, adj, tm, softmax_dim=1):
        if tm==True: we=torch.tensor(0.5).to('cuda')
        else: we=torch.tensor(0.5)

        p = F.normalize(s, dim=1, p=2)

        p = p.reshape(p.size(0),-1) 

        p = self.bnpi(p)
        p = F.prelu(p, we)

        p = self.state_pi(p)

        p = self.state_pi_bn(p)
        p = F.prelu(p, we)

        p = self.fcp_1(p)

        p = self.fcp_bn1(p)
        p = F.prelu(p, we)

        p = self.fcp_2(p)
        p = self.fcp_bn2(p)
        p = F.prelu(p, we)

        p = self.fcp(p)
        return F.softmax(p, dim=softmax_dim)


    def temp(self, s, adj):
        t = F.normalize(s, dim=1, p=2)

        t = t.reshape(t.size(0),-1) 

        t = self.state_temp(t)
        t = self.state_temp_bn(t)
        t = F.relu(t)

        t = self.fctemp_1(t)
        t = self.fctemp_bn1(t)
        t = F.relu(t)

        t = self.fctemp_2(t)
        t = self.fctemp_bn2(t)
        t = F.relu(t)

        t = self.fctemp(t)
        return F.relu(t)


    def put_data(self, transition):
        ############### RECEIVE DATA
        self.data.append(transition)
    
    def make_batch(self):
        ############### DATA BATCH
        s_lst, a_lst, r_lst, s_prime_lst, done_lst, adj_lst, adj_prime_lst = [], [], [], [], [], [], []
        for transition in self.data:
            ############### ADD DATA
            s,a,r,s_prime,done,adj,adj_prime = transition
            s_lst.append(s.clone().cpu().numpy().reshape(-1,116,116)) #np
            a_lst.append([a]) # list
            r_lst.extend([r])
            s_prime_lst.append(s_prime.clone().cpu().numpy().reshape(-1,116,116))
            done_mask = 0.0 if done else 1.0
            done_lst.append(done_mask)
            adj_lst.append(adj.clone().cpu().numpy().reshape(-1,116,116))
            adj_prime_lst.append(adj_prime.clone().cpu().numpy().reshape(-1,116,116))

        ############### TO TENSOR
        s_batch, a_batch, r_batch, s_prime_batch, done_batch, adj_batch, adj_prime_batch, \
            = torch.tensor(s_lst, dtype=torch.float, requires_grad=True, device=torch.device('cuda:0')),\
            torch.tensor(a_lst, dtype=torch.float, requires_grad=True, device=torch.device('cuda:0')), \
            torch.tensor(r_lst, dtype=torch.float, requires_grad=True, device=torch.device('cuda:0')),\
            torch.tensor(s_prime_lst, dtype=torch.float, requires_grad=True, device=torch.device('cuda:0')), \
            torch.tensor(done_lst, dtype=torch.float, requires_grad=True, device=torch.device('cuda:0')),\
            torch.tensor(adj_lst, dtype=torch.float, requires_grad=True, device=torch.device('cuda:0')),\
            torch.tensor(adj_prime_lst, dtype=torch.float, requires_grad=True, device=torch.device('cuda:0'))
        self.data = []

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch, adj_batch, adj_prime_batch

    def train_net(self, model, target_model, model2, target_model2, iters):

        scaler = torch.cuda.amp.GradScaler()
        _, _, _, lr, wd,  _, _, _, _ = util.load_hyperparameter()

        ############### OPTIMIZER
        if args.optimizer=='RAdam':
            self.optimizer = optim.RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd, betas=args.betas)
            self.optimizer2 = optim.RAdam(filter(lambda p: p.requires_grad, model2.parameters()), lr=lr, weight_decay=wd, betas=args.betas)

        if args.optimizer=='adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd, betas=args.betas)
            self.optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model2.parameters()), lr=lr, weight_decay=wd, betas=args.betas)

        if args.optimizer=='sgd':
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=lr, weight_decay=wd, momentum=args.momentum)
            self.optimizer2 = torch.optim.SGD(filter(lambda p: p.requires_grad, model2.parameters()),lr=lr, weight_decay=wd, momentum=args.momentum)

        if args.optimizer=='rmsprop':
            self.optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),lr=lr, weight_decay=wd, momentum=args.momentum)
            self.optimizer2 = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model2.parameters()),lr=lr, weight_decay=wd, momentum=args.momentum)

        if args.optimizer=='madgrad':
            self.optimizer = madgrad.MADGRAD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd, momentum=args.momentum)
            self.optimizer2 = madgrad.MADGRAD(filter(lambda p: p.requires_grad, model2.parameters()), lr=lr, weight_decay=wd, momentum=args.momentum)

        if args.optimizer=='adamw':
            self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd, eps=1e-8, betas=args.betas)
            self.optimizer2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model2.parameters()), lr=lr, weight_decay=wd, eps=1e-8, betas=args.betas)

        ############### SCHEDULER
        scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        scheduler2 = lr_scheduler.ExponentialLR(self.optimizer2, gamma=0.99)

        ############### BATCH DATA
        s, a, r, s_prime, done, adj, adj_prime = self.make_batch()

        ############### LOADER
        s_dataset = TensorDataset(s, a, r, s_prime, done, adj, adj_prime)
        s_loader = DataLoader(s_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, drop_last=True)

        for s_batch, a_batch, r_batch, s_prime_batch, done_batch, adj_batch, adj_prime_batch in s_loader:  
            mask = adj_batch 
            mask_prime = adj_prime_batch

            ############### CUDA
            if use_cuda:
                s_batch, a_batch, r_batch, s_prime_batch, done_batch, mask, mask_prime\
                    =  s_batch.cuda(), a_batch.cuda(), r_batch.cuda(),\
                        s_prime_batch.cuda(), done_batch.cuda(),\
                        mask.cuda(), mask_prime.cuda()

            ############### ZERO GRAD
            self.optimizer.zero_grad()
            self.optimizer2.zero_grad()

            action_s = torch.tensor([[x.item()] for x in a_batch], device=torch.device('cuda:0'), dtype=int)

            ############### s_prime -> a_prime 
            action_prime = [] 
            for p in util.PI(model, s_prime_batch, mask_prime,True):
                scores = util.softmax(p.clone().detach().cpu())
                action_prime.append([np.random.choice(np.arange(args.action), p=scores)])
            action_prime =torch.tensor(action_prime, device=torch.device('cuda:0'), dtype=int)

            with torch.cuda.amp.autocast():    # cast mixed precision

                PI = util.PI(model, s_batch, mask, True)
                PI = torch.gather(PI, 1, action_s)

                PI_PRIME = util.PI(model, s_prime_batch, mask_prime, True)
                PI_PRIME = torch.gather(PI_PRIME, 1, action_prime)

                TEMP = util.T(model, s_batch.cuda(), mask, True)
                Q1 = util.Q(model, s_batch.cuda(), action_s.cuda(), mask, True)

                target_Q1 = util.Q(target_model, s_prime_batch.cuda(), action_prime.cuda(), mask_prime, True)
                target_Q2 = util.Q(target_model2, s_prime_batch.cuda(), action_prime.cuda(), mask_prime, True)

                target_Q = torch.minimum(target_Q1, target_Q2)
                Q_value = Q1

                Q_detach = Q_value.clone().detach()
                TARGET_Q_detach = target_Q.clone().detach()
                PI_detach = PI.clone().detach()
                PI_PRIME_detach = PI.clone().detach()
                TEMP_detach = torch.tensor(TEMP).clone().detach()

                
                ############### LOSS
                TD_target = (r_batch + gamma * (target_Q + TEMP_detach * torch.log10(PI_PRIME_detach)) * done_batch)
                soft_q1_value_error =  F.smooth_l1_loss(Q1, TD_target)

                soft_policy_error =  (TEMP_detach * torch.log10(PI) - Q_detach) ##################
                temperature_obj = (-TEMP * (torch.log10(PI_detach) + torch.tensor(args.target_entropy)))

                ############### OBJECTIVE FUNCTIONS
                soft_policy_error = (args.policy * soft_policy_error).mean()
                soft_q1_value_error = (args.value * soft_q1_value_error).mean()
                temperature_error = (args.temp * temperature_obj).mean()

                objective_function = soft_policy_error + soft_q1_value_error + temperature_error

                scaler.scale(soft_q1_value_error).backward(retain_graph=True)

                scaler.scale(soft_policy_error).backward()
                scaler.scale(temperature_error).backward()

                ############### MODEL PARAMETER CLAMP
                if args.w_clamp:
                    for param_tensor in model.state_dict():
                        if param_tensor.find('weight'):
                            model.state_dict()[param_tensor].clamp_(-args.w_clamp_v, args.w_clamp_v)
                    for param_tensor in model2.state_dict():
                        if param_tensor.find('weight'):
                            model2.state_dict()[param_tensor].clamp_(-args.w_clamp_v, args.w_clamp_v)

                if args.g_clamp:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.g_clamp_v)
                    torch.nn.utils.clip_grad_norm_(model2.parameters(), args.g_clamp_v)

                ############### UPDATE
                scaler.step(self.optimizer)   # unscaled gradients
                scaler.update() 

                scaler.step(self.optimizer2)   # unscaled gradients
                scaler.update() 

                scheduler.step()
                scheduler2.step()

            ################ MODEL SAVE
            torch.save(model, args.dualnetwork_model_init_path)
            torch.save(model2, args.dualnetwork_model_init2_path)

        wandb.log({
            'Soft Policy': soft_policy_error,
            'Soft Value': soft_q1_value_error,
            'Temperature': temperature_error
            })
        
        val_es = objective_function

        #obj, sopi = objective_function, soft_policy_error

        return soft_policy_error, soft_q1_value_error, 0, temperature_error, 0, val_es

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    import seed
    seed.seed_everything(args.seed)
    from multiprocessing import freeze_support
    freeze_support()
    
    net_x = ActorCritic(in_features=args.in_feature, out_features=args.out_feature, K=args.k, weight_init='xavier') 
    net_k = ActorCritic(in_features=args.in_feature, out_features=args.out_feature, K=args.k, weight_init='kaimimg') 

    with open('path.txt', mode='wt', encoding='utf-8') as f:
        f.writelines('{}'.format(os.getcwd()))

    path=args.path
    torch.save(net_x, path + '/train_dual_network/best.pt')
    torch.save(net_k, path + '/train_dual_network/best2.pt')
    torch.save(net_x, path + '/train_dual_network/target.pt')
    torch.save(net_k, path + '/train_dual_network/target2.pt')
    torch.save(net_x, path + '/train_dual_network/arXiv/origin.pt')
    torch.save(net_k, path + '/train_dual_network/arXiv/origin2.pt')
