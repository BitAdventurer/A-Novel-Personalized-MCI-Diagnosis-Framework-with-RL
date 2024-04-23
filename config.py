import argparse
from datetime import datetime

# Initialize parser
parser = argparse.ArgumentParser(description='Configuration for the training environment')

# Reading path from a file safely
try:
    with open('path.txt', mode='rt', encoding='utf-8') as file:
        path = file.readline().strip()
except FileNotFoundError:
    print("path.txt not found. Ensure the file is in the current directory.")
    path = ""

# Timestamp for filenames or logging
timestamp = datetime.today().strftime("_%Y%m%d%H%M%S")

# Base Paths
parser.add_argument("--path", default=path, type=str)
parser.add_argument("--dualnetwork_best_path", default=f'{path}/train_dual_network/best.pt', type=str)
parser.add_argument("--dualnetwork_best2_path", default=f'{path}/train_dual_network/best2.pt', type=str)
parser.add_argument("--dualnetwork_target_path", default=f'{path}/train_dual_network/target.pt', type=str)
parser.add_argument("--dualnetwork_target2_path", default=f'{path}/train_dual_network/target2.pt', type=str)
parser.add_argument("--dualnetwork_model_init_path", default=f'{path}/train_dual_network/arXiv/origin.pt', type=str)
parser.add_argument("--dualnetwork_model_init2_path", default=f'{path}/train_dual_network/arXiv/origin2.pt', type=str)

# Environment Parameters
parser.add_argument("--stop_point", default=115, type=int)
parser.add_argument("--sp_game_count", default=10, type=int)  # Updated with an example default
parser.add_argument("--num_process", default=1, type=int)    # Updated with an example default
parser.add_argument("--temperature", default=1, type=int)    # Updated with an example default
parser.add_argument("--action", default=117, type=int)
parser.add_argument("--reward", default=1.0, type=float)
parser.add_argument("--seed", default=42, type=int)          # Updated with an example default
parser.add_argument("--num_epoch", default=10, type=int)     # Updated with an example default

# Training Hyperparameters
parser.add_argument("--lr", default=0.005, type=float)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--wd", default=0.00001, type=float)
parser.add_argument("--betas", default=(0.9, 0.999), type=tuple)
parser.add_argument("--momentum", default=0.9, type=float)   # Updated with an example default
parser.add_argument("--gamma", default=0.9, type=float)
parser.add_argument("--discount_factor", default=0.95, type=float)
parser.add_argument("--epochs", default=100, type=int)       # Updated with an example default
parser.add_argument("--iters", default=1000, type=int)       # Updated with an example default
parser.add_argument("--k", default=3, type=int)
parser.add_argument("--in_feature", default=116, type=int)
parser.add_argument("--out_feature", default=8, type=int)

# Actor-Critic Layers (placeholders for actual default values)
parser.add_argument("--actor_layer1", default=64, type=int)  # Updated with an example default
parser.add_argument("--critic_layer1", default=64, type=int)  # Updated with an example default

# Buffer and Device Configuration
parser.add_argument("--buffer_size", default=1000, type=int)  # Updated with an example default
parser.add_argument("--cuda_device", default=0, type=int)

# Optional feature toggles
parser.add_argument("--feature", default=True, type=bool)

# Parse arguments
args = parser.parse_args()
