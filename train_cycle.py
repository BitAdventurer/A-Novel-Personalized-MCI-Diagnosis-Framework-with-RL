import os
import torch
from random import random
from tqdm import tqdm
import warnings

# Custom module imports
import config
import util
import seed
import train_network
import loss_idd_dual
import early_stop_dual_main
import evaluate_best_player_val_p
from multiprocessing import freeze_support
import wandb

# Suppress warnings and login to wandb
warnings.filterwarnings("ignore")
wandb.login()

# Parsing arguments
args = config.parser.parse_args()
base_path = args.path
torch.set_num_threads(os.cpu_count() * 2)

def cleanup_replay_buffer(validation_list):
    """Ensure the replay buffer size is maintained within limits."""
    backup_path = os.path.join(base_path, 'self_play_backup')
    buffer_files = sorted(os.listdir(backup_path))  # Sort to remove the oldest first

    # Remove oldest files if buffer is full
    while len(buffer_files) > args.buffer_size:
        os.remove(os.path.join(backup_path, buffer_files.pop(0)))

    # Fill up the replay buffer if below desired size
    while len(buffer_files) < args.buffer_size:
        util.self_play(validation_list)
        buffer_files = os.listdir(backup_path)

def update_hyperparameters():
    """Adjust hyperparameters based on certain conditions."""
    args.temperature *= 0.9999
    args.lr *= 0.9999
    if random() < 0.5:
        args.stop_point += 0  # Example modification

    # Update wandb configuration
    wandb.config.update(dict(temperature=args.temperature, lr=args.lr,
                             stop_point=args.stop_point, momentum=args.momentum,
                             soft_update_ratio=args.soft_update_ratio), allow_val_change=True)

def main_cycle():
    """Main training and validation cycle."""
    early_stopping = early_stop_dual_main.EarlyStopping(patience=50, verbose=True)
    validation_data = util.validation_data(args.split)
    validation_list = list(range(len(validation_data)))

    for i in tqdm(range(args.iters), desc="Iteration Progress"):
        wandb.init(project=args.new_server, name=f'Iteration {i}', reinit=True)
        wandb.config.update(vars(args), allow_val_change=True)

        cleanup_replay_buffer(validation_list)
        seed.seed_everything(args.seed)
        
        train_network.train_main(True, i)  # Train and assume success
        loss_idd_dual.loss_idd(i)
        early_stopping_point, agent_vs = evaluate_best_player_val_p.confusion(i, False)
        
        if early_stopping(early_stopping_point):
            print("Early stopping triggered.")
            break

        if agent_vs:  # If current agent is better
            update_hyperparameters()

        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    seed.seed_everything(args.seed)
    freeze_support()
    main_cycle()
