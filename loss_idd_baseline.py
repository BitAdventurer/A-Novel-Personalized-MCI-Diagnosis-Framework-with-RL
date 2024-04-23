import pandas as pd
import matplotlib.pyplot as plt
import config

# Configure the script parameters
args = config.parser.parse_args()
base_path = args.path
fold_number = args.fold

def load_loss_data(file_path):
    """
    Load loss data from a CSV file.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_path} is empty.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def plot_loss(data, save_path):
    """
    Plot train, validation, and test loss from the loaded data.
    """
    if data is None:
        print("No data available to plot.")
        return

    plt.plot(data['0'], label='Train', marker='o', ls='-', markersize=2)
    plt.plot(data['0.1'], label='Val', marker='o', ls='-', markersize=2)
    plt.plot(data['0.2'], label='Test', marker='o', ls='-', markersize=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=300)
    plt.show()

def loss_idd(iteration):
    """
    Function to manage loading and plotting loss data for a specific iteration.
    """
    file_path = f'{base_path}/baseline/fold{fold_number}/baseline_loss{iteration}.csv'
    save_path = f'{base_path}/baseline/fold{fold_number}/loss.png'
    
    data = load_loss_data(file_path)
    plot_loss(data, save_path)

if __name__ == "__main__":
    iteration = 1  # Example iteration number, this should be parameterized as needed.
    loss_idd(iteration)
