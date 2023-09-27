import argparse
# from argparse import ArgumentParser

from datetime import datetime
from os import truncate

parser = argparse.ArgumentParser(description='Argparse')
timestamp = datetime.today().strftime("_%Y%m%d%H%M%S")

with open('path.txt', mode='rt', encoding='utf-8') as f:
    lines = f.readlines()  
    path = str(lines[0])

# PATH
parser.add_argument("--path", default=path, type=str)
parser.add_argument("--dualnetwork_best_path", default=path + '/train_dual_network/best.pt', type=str, help="best.pt")
parser.add_argument("--dualnetwork_best2_path", default=path + '/train_dual_network/best2.pt', type=str, help="best2.pt")
parser.add_argument("--dualnetwork_target_path", default=path + '/train_dual_network/target.pt', type=str, help="target.pt")
parser.add_argument("--dualnetwork_target2_path", default=path + '/train_dual_network/target2.pt', type=str, help="target2.pt")
parser.add_argument("--dualnetwork_model_init_path", default=path + '/train_dual_network/arXiv/origin.pt', type=str, help="origin.pt")
parser.add_argument("--dualnetwork_model_init2_path", default=path + '/train_dual_network/arXiv/origin2.pt', type=str, help="origin2.pt")

# environment parameter
parser.add_argument("--stop_point", default=115, type=int, help="Terminal point")
parser.add_argument("--sp_game_count", default=, type=int, help="N of self play")
parser.add_argument("--num_process", default=, type=int, help="N of multiprocess")
parser.add_argument("--temperature", default=, type=int, help="Temperature")
parser.add_argument("--action", default=117, type=int, help="N of action")
parser.add_argument("--reward", default=1.0, type=float, help="Reward")
parser.add_argument('--seed', default=, type=int, help='random seed')
parser.add_argument('--num_epoch', default=, type=int, help='num_epoch')
parser.add_argument('--one_hot', default=False, type=str, help='One_hot encoding')

# training hyperparams
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument("--batch_size", default=32, type=int, help="batch size of Actor-Critic")
parser.add_argument('--wd', default=0.00001, type=float, help='weight decay')
parser.add_argument('--betas', default=(0.9, 0.999), type=tuple, help='adam betas')
parser.add_argument("--momentum", default=, type=float, help="optimizer momentum - SGD, MADGRAD")
parser.add_argument("--gamma", default=0.9, type=float, help="gamma for lr learning")
parser.add_argument("--discount_factor", default=0.95, type=float, help="Discount factor")
parser.add_argument("--epochs", default=, type=int, help="Actor-Critic train epochs")
parser.add_argument("--iters", default=, type=int, help="iteration")
parser.add_argument("--k", default=3, type=int, help="K of baseline")
parser.add_argument("--in_feature", default=116, type=int, help="Infeature")
parser.add_argument("--out_feature", default=8, type=int, help="Outfeature")
parser.add_argument("--target_entropy", default=0.5, type=float, help="Loss entropy")
parser.add_argument("--policy", default=1.0, type=float, help="Loss policy")
parser.add_argument("--value", default=1.0, type=float, help="Loss value")
parser.add_argument("--temp", default=1.0, type=float, help="Loss temperature")
parser.add_argument("--optimizer", default='sgd', type=str, help="Actor-Critic optimizer RAdam/madgrad/sgd/adam/rmsprop/adamw")
parser.add_argument("--patience", default=, type=int, help="Early stop point")
parser.add_argument("--window", default=, type=int, help="Sampling window size")

# actor-critic
parser.add_argument("--actor_layer1", default=512, type=int, help="Actor network layer1")
parser.add_argument("--actor_layer2", default=256, type=int, help="Actor network layer2")
parser.add_argument("--actor_layer", default=117, type=int, help="Actor network layer6")

parser.add_argument("--critic_layer1", default=64, type=int, help="Critic network layer1")
parser.add_argument("--critic_layer2", default=32, type=int, help="Critic network layer2")
parser.add_argument("--critic_layer", default=1, type=int, help="Critic network layer6")

parser.add_argument("--temp_layer1", default=64, type=int, help="Actor network layer1")
parser.add_argument("--temp_layer2", default=32, type=int, help="Actor network layer2")
parser.add_argument("--temp_layer", default=1, type=int, help="Actor network layer6")

parser.add_argument("--action_emb_layer", default=32, type=int, help="Action emb layer")
parser.add_argument("--state_emb_layer", default=32, type=int, help="State emb layer")

# buffer
parser.add_argument("--self_data", default=path+'/self_play_best_data', type=str, help="Generated data")
parser.add_argument("--replay_buffer", default=True, type=bool, help="Replay buffer use")
parser.add_argument("--buffer_data", default=path+'/self_play_backup', type=str, help="Generated data path")
parser.add_argument("--buffer_size", default=1000, type=int, help="Buffer_size")
parser.add_argument("--sampling_size", default=2, type=int, help="Sampling_size")

# tagerget network
parser.add_argument("--target_network", default=True, type=bool, help="Used target network ")
parser.add_argument("--soft_update", default=False, type=bool, help="Soft update")
parser.add_argument("--soft_update_ratio", default=0.01, type=float, help="Soft update ratio")
parser.add_argument("--target_n", default=1, type=int, help="Target network freeze N")
parser.add_argument("--target_soft_update_ratio", default=0.01, type=float, help="Target soft update ratio")

# data
fold = 0 #fold1-5
if fold == 0:
    parser.add_argument("--split", default=1, type=int, help="Select N of split")
    parser.add_argument("--fold", default=0, type=int, help="N of fold")
    parser.add_argument("--new_server", default='SAC_X', type=str, help="Device name")
    parser.add_argument("--plt", default=True, type=bool, help="Plot")


parser.add_argument("--n_split", default=5, type=int, help="N")
parser.add_argument("--jeilt", default=0.0, type=float, help="Threshold")

# device
parser.add_argument("--cuda_device", default=0, type=int, help="Cuda")

# Clamp
parser.add_argument("--w_clamp", default=False, type=bool, help="Weight clamp")
parser.add_argument("--w_clamp_v", default=1, type=float, help="Weight clamp value")
parser.add_argument("--g_clamp", default=False, type=bool, help="Grandient clamp")
parser.add_argument("--g_clamp_v", default=2, type=float, help="Grandient clamp value")

# Plot
parser.add_argument("--eval", default=True, type=bool, help="evaluate")
parser.add_argument("--val_eval", default=False, type=bool, help="evaluate")
parser.add_argument("--feature", default=True, type=bool, help="Critic feature")

# test
parser.add_argument("--state_value", default=True, type=bool, help="State value")
