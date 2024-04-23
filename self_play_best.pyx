import os
import numpy as np
import torch
import warnings
import config
import time
import util
from pathlib import Path

warnings.filterwarnings("ignore")
args = config.parser.parse_args()

# Paths and constants
DATA_PATH = Path(args.path)
ACTOR_CRITIC_PATH = args.dualnetwork_best_path
LABEL_TRUE = [1, 0]  # Assuming NC = 1, MCI = 0
LABEL_FALSE = [0, 1]
REST = 116

# Utility function to seed everything for reproducibility
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Load the actor-critic model
def load_model(model_path):
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    return model

# Get the game count based on the subject's label
def get_game_count(label, nc_multiplier=1.5):
    return int(args.sp_game_count * nc_multiplier) if label == LABEL_TRUE else args.sp_game_count

# Write initial success index
def initialize_success_index(subj, idx):
    idx_path = DATA_PATH / f'terminal/val/succes_idx/{subj}_{idx}.txt'
    if not os.path.exists(idx_path):
        with open(idx_path, 'wt', encoding='utf-8') as file:
            file.write('0')

# Simulation of one episode
def simulate_episode(subj, stop_point, temperature, model, classifier):
    cross_validation = util.validation_data(args.split)
    cross_validation_label = util.validation_label(args.split)
    data = np.array(cross_validation[subj], dtype=float).reshape(1, 1, 116, 116)
    x0 = torch.tensor(data, dtype=torch.float32)

    # Setup initial state and environment
    state = util.State(data, subj, False)
    adj_state = util.State(torch.ones(116, 116), subj, True)

    true_label = LABEL_TRUE if cross_validation_label[subj] == LABEL_FALSE else LABEL_FALSE
    input_data = classifier(x0)
    _, predicted_label = torch.max(input_data, dim=1)

    history = []

    # Simulation loop
    for step in range(stop_point):
        if true_label == predicted_label.tolist():
            update_history(history, state, adj_state, args.reward)
            break

        scores = util.pi_network(model, state.piece, adj_state.piece, temperature)
        action = np.random.choice(116, p=scores)
        state = state.next(action)
        adj_state = adj_state.next(action)

        history.append([state.piece, action, None, None, None, None, None])
        temperature *= 0.9  # Cool down

        # Update state in the history
        update_history(history, state, adj_state, args.reward, terminal=(step == stop_point - 1))

    return history, true_label == predicted_label.tolist()

# Update the history with rewards and transitions
def update_history(history, state, adj_state, reward, terminal=False):
    for i, entry in enumerate(history):
        entry[2] = reward * ((args.discount_factor ** i) if not terminal else 1)
        entry[3] = state if i == len(history) - 1 else history[i + 1][0]
        entry[4] = 1 if not terminal else 0
        entry[5] = adj_state if i == 0 else history[i - 1][6]
        entry[6] = adj_state

# Main function to control the flow of the self-play simulations
def self_play(subj):
    start_time = time.time()
    stop_point, temperature = util.load_hyperparameters()
    seed_everything(int(time.time() * 1000.0))

    label = util.validation_label(args.split)[subj]
    game_count = get_game_count(label)
    actor_critic = load_model(ACTOR_CRITIC_PATH)
    classifier = util.classifier()

    for idx in range(game_count):
        initialize_success_index(subj, idx)
        history, termination_flag = simulate_episode(subj, stop_point, temperature, actor_critic, classifier)
        util.best_write_data(history, subj, start_time, idx)
        if termination_flag:
            break

    seed_everything(args.seed)

if __name__ == "__main__":
    subject_id = 0  # Example subject ID
    self_play(subject_id)
