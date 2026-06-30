"""Command-line configuration shared by the MCI-RL experiment scripts.

The original research code reads most paths and hyperparameters from a global
``args`` object. Keeping that interface avoids broad changes across the Cython
modules while still centralizing the available options in one place.
"""

import argparse
from datetime import datetime
from pathlib import Path


def str2bool(value):
    """Parse common string forms for boolean command-line flags."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def read_default_path():
    """Read the local data root from a repo-level path configuration file."""
    # Keep the old root-level path.txt fallback for compatibility with earlier
    # clones, while preferring the organized configs/ location.
    candidates = (Path("configs/path.txt"), Path("path.txt"))
    for path_file in candidates:
        try:
            return path_file.read_text(encoding="utf-8").splitlines()[0].strip()
        except (FileNotFoundError, IndexError):
            continue
    return ""


DEFAULT_PATH = read_default_path()
DEFAULT_TIMESTAMP = datetime.today().strftime("_%Y%m%d%H%M%S")

parser = argparse.ArgumentParser(
    description="Configuration for sparse graph RL-based MCI diagnosis"
)

# Paths
parser.add_argument("--path", default=DEFAULT_PATH, type=str, help="Local project/data root")
parser.add_argument("--dualnetwork_best_path", default=f"{DEFAULT_PATH}/train_dual_network/best.pt", type=str)
parser.add_argument("--dualnetwork_best2_path", default=f"{DEFAULT_PATH}/train_dual_network/best2.pt", type=str)
parser.add_argument("--dualnetwork_target_path", default=f"{DEFAULT_PATH}/train_dual_network/target.pt", type=str)
parser.add_argument("--dualnetwork_target2_path", default=f"{DEFAULT_PATH}/train_dual_network/target2.pt", type=str)
parser.add_argument("--dualnetwork_model_init_path", default=f"{DEFAULT_PATH}/train_dual_network/arXiv/origin.pt", type=str)
parser.add_argument("--dualnetwork_model_init2_path", default=f"{DEFAULT_PATH}/train_dual_network/arXiv/origin2.pt", type=str)
parser.add_argument("--self_data", default=f"{DEFAULT_PATH}/self_play_best_data", type=str)
parser.add_argument("--buffer_data", default=f"{DEFAULT_PATH}/self_play_backup", type=str)
parser.add_argument("--best_path", default=f"{DEFAULT_PATH}/self_play_best_data", type=str)
parser.add_argument("--backup_path", default=f"{DEFAULT_PATH}/self_play_backup", type=str)

# Dataset and split settings
parser.add_argument("--fold", default=0, type=int)
parser.add_argument("--split", default=0, type=int)
parser.add_argument("--n_split", default=5, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--dataset_seed", default=42, type=int)

# Environment parameters
parser.add_argument("--stop_point", default=115, type=int)
parser.add_argument("--sp_game_count", default=10, type=int)
parser.add_argument("--num_process", default=1, type=int)
parser.add_argument("--temperature", default=1.0, type=float)
parser.add_argument("--action", default=117, type=int)
parser.add_argument("--reward", default=1.0, type=float)
parser.add_argument("--num_epoch", default=10, type=int)
parser.add_argument("--state_value", default=False, type=str2bool)
parser.add_argument("--jeilt", default=False, type=str2bool)

# Training hyperparameters
parser.add_argument("--lr", default=0.005, type=float)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--wd", default=0.00001, type=float)
parser.add_argument("--betas", default=(0.9, 0.999), type=tuple)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--gamma", default=0.9, type=float)
parser.add_argument("--discount_factor", default=0.95, type=float)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--iters", default=1000, type=int)
parser.add_argument("--patience", default=50, type=int)
parser.add_argument("--sampling_size", default=32, type=int)
parser.add_argument("--window", default=100, type=int)
parser.add_argument("--target_n", default=10, type=int)
parser.add_argument("--soft_update_ratio", default=0.005, type=float)
parser.add_argument("--target_soft_update_ratio", default=0.005, type=float)
parser.add_argument("--soft_update", default=True, type=str2bool)
parser.add_argument("--optimizer", default="Adam", type=str)

# Graph/network dimensions
parser.add_argument("--k", default=3, type=int)
parser.add_argument("--in_feature", default=116, type=int)
parser.add_argument("--out_feature", default=8, type=int)
parser.add_argument("--state_emb_layer", default=256, type=int)
parser.add_argument("--action_emb_layer", default=117, type=int)
parser.add_argument("--actor_layer1", default=128, type=int)
parser.add_argument("--actor_layer2", default=64, type=int)
parser.add_argument("--actor_layer", default=117, type=int)
parser.add_argument("--critic_layer1", default=128, type=int)
parser.add_argument("--critic_layer2", default=64, type=int)
parser.add_argument("--critic_layer", default=1, type=int)
parser.add_argument("--temp_layer1", default=128, type=int)
parser.add_argument("--temp_layer2", default=64, type=int)
parser.add_argument("--temp_layer", default=1, type=int)

# Loss weights and optimization options
parser.add_argument("--policy", default=1.0, type=float)
parser.add_argument("--value", default=1.0, type=float)
parser.add_argument("--temp", default=1.0, type=float)
parser.add_argument("--target_entropy", default=-1.0, type=float)
parser.add_argument("--one_hot", default=False, type=str2bool)
parser.add_argument("--feature", default=True, type=str2bool)
parser.add_argument("--replay_buffer", default=True, type=str2bool)
parser.add_argument("--val_eval", default=True, type=str2bool)
parser.add_argument("--w_clamp", default=False, type=str2bool)
parser.add_argument("--g_clamp", default=False, type=str2bool)
parser.add_argument("--g_clamp_v", default=1.0, type=float)

# Device and logging
parser.add_argument("--buffer_size", default=1000, type=int)
parser.add_argument("--cuda_device", default=0, type=int)
parser.add_argument("--new_server", default="mci-rl", type=str)
parser.add_argument("--timestamp", default=DEFAULT_TIMESTAMP, type=str)
parser.add_argument("--plt", default=False, type=str2bool)

args = parser.parse_args()
