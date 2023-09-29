import os
import torch
import wandb
import config
import util
import seed
import reward
import train_network
import loss_idd_dual
import early_stop_dual_main
import evaluate_best_player_val_p
from random import random
from tqdm import tqdm
from multiprocessing import freeze_support

# Set Environment and Global Configurations
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.filterwarnings("ignore")
wandb.login()

args = config.parser.parse_args()
base_path = args.path
torch.set_num_threads(os.cpu_count() * 2)

def cleanup_replay_buffer(validation_list):
    """Remove oldest files if replay buffer is full, and fills the replay buffer if not."""
    backup_path = os.path.join(base_path, 'self_play_backup')
    buffer_files = os.listdir(backup_path)

    if len(buffer_files) >= args.buffer_size:
        to_remove = buffer_files[:args.sp_game_count * len(validation_list)]
        for file in to_remove:
            os.remove(os.path.join(backup_path, file))

    while len(os.listdir(backup_path)) < args.buffer_size:
        util.self_play(validation_list)


def update_hyperparameters():
    """Update and Write the hyperparameters."""
    args.temperature *= 0.9999
    args.lr *= 0.9999
    args.soft_update_ratio *= 1.0
    if random() <= 0.5:
        args.stop_point += 0

    wandb.config.update(args, allow_val_change=True)
    util.hyperparameter(args.stop_point, args.temperature, args.sp_game_count, args.lr, args.wd,
                        args.momentum, args.soft_update_ratio, args.batch_size, args.buffer_size)


def main_cycle():
    early_stopping = early_stop_dual_main.EarlyStopping(patience=50, verbose=True, delta=0.0)
    validation_data = util.validation_data(args.split)
    validation_list = list(range(len(validation_data)))

    util.hyperparameter(args.stop_point, args.temperature, args.sp_game_count, args.lr, args.wd,
                        args.momentum, args.soft_update_ratio, args.batch_size, args.buffer_size)

    util.init_performance([(j, 999) for j in range(len(validation_data))], -1)

    for i in tqdm(range(args.iters)):
        wandb.init(project=args.new_server, name=f'{i}', allow_val_change=True, reinit=True)
        wandb.config.update(args, allow_val_change=True)

        cleanup_replay_buffer(validation_list)
        seed.seed_everything(args.seed)

        ############### REPLAY BUFFER
        while len(os.listdir(args.path+'/self_play_backup')) < args.buffer_size:
            util.self_play(validation_list)

        train_network.train_main(True, i)  # assuming result is always True
        reward.plot_acc_val()
        loss_idd_dual.loss_idd(i)

        early_stopping_point, agent_vs = evaluate_best_player_val_p.confusion(i, False)
        early_stopping(early_stopping_point)

        if agent_vs:  # current agent win
            update_hyperparameters()

        torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    seed.seed_everything(args.seed)
    freeze_support()
    main_cycle()
