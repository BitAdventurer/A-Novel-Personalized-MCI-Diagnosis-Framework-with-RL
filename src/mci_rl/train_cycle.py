"""Main reinforcement-learning training loop for sparse FCN construction."""

import os
import warnings
from multiprocessing import freeze_support
from random import random

import torch
import wandb
from tqdm import tqdm

from mci_rl import config
from mci_rl import early_stop_dual_main
from mci_rl import evaluate_best_player_val_p
from mci_rl import loss_idd_dual
from mci_rl import seed
from mci_rl import train_network
from mci_rl import util


args = config.parser.parse_args()
base_path = args.path


def configure_runtime():
    """Set process-level runtime options before training starts."""
    warnings.filterwarnings("ignore")
    wandb.login()
    torch.set_num_threads(os.cpu_count() * 2)


def cleanup_replay_buffer(validation_list):
    """Keep replay-buffer files within ``args.buffer_size``."""
    backup_path = os.path.join(base_path, "self_play_backup")
    buffer_files = sorted(os.listdir(backup_path))

    while len(buffer_files) > args.buffer_size:
        oldest_file = buffer_files.pop(0)
        os.remove(os.path.join(backup_path, oldest_file))

    while len(buffer_files) < args.buffer_size:
        util.self_play(validation_list)
        buffer_files = sorted(os.listdir(backup_path))


def update_hyperparameters():
    """Apply lightweight schedules after the current agent improves."""
    args.temperature *= 0.9999
    args.lr *= 0.9999

    # Retained from the original implementation; this hook is a no-op unless
    # future experiments decide to modify stop_point probabilistically.
    if random() < 0.5:
        args.stop_point += 0

    wandb.config.update(
        {
            "temperature": args.temperature,
            "lr": args.lr,
            "stop_point": args.stop_point,
            "momentum": args.momentum,
            "soft_update_ratio": args.soft_update_ratio,
        },
        allow_val_change=True,
    )


def run_iteration(iteration, validation_list):
    """Run one self-play, training, loss, and validation cycle."""
    wandb.init(project=args.new_server, name=f"Iteration {iteration}", reinit=True)
    wandb.config.update(vars(args), allow_val_change=True)

    cleanup_replay_buffer(validation_list)
    seed.seed_everything(args.seed)

    train_network.train_main(True, iteration)
    loss_idd_dual.loss_idd(iteration)
    return evaluate_best_player_val_p.confusion(iteration, False)


def main_cycle():
    """Train until validation stops improving or ``args.iters`` is reached."""
    early_stopping = early_stop_dual_main.EarlyStopping(patience=50, verbose=True)
    validation_data = util.validation_data(args.split)
    validation_list = list(range(len(validation_data)))

    for iteration in tqdm(range(args.iters), desc="Iteration Progress"):
        early_stopping_point, agent_improved = run_iteration(iteration, validation_list)

        if early_stopping(early_stopping_point):
            print("Early stopping triggered.")
            break

        if agent_improved:
            update_hyperparameters()

        torch.cuda.empty_cache()


if __name__ == "__main__":
    configure_runtime()
    torch.multiprocessing.set_start_method("spawn", force=True)
    seed.seed_everything(args.seed)
    freeze_support()
    main_cycle()
