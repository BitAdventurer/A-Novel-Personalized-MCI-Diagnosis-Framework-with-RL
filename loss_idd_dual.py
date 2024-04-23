import pandas as pd
import matplotlib.pyplot as plt
import config
import gc
from datetime import datetime

args = config.parser.parse_args()
path = args.path

def read_loss_data(file_path):
    """
    Reads loss data from a CSV file and handles potential errors.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame()  # Return empty DataFrame if file not found
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()

def plot_loss(data, title, color, subplot_position):
    """
    Plots individual loss data on a specific subplot.
    """
    if data.empty:
        print(f"No data to plot for {title}.")
        return

    plt.subplot(1, 3, subplot_position)
    plt.title(title)
    plt.plot(data['0'], label=title, marker='o', lw=2, ls='-', markersize=2, color=color)
    plt.legend()
    plt.grid()
    plt.tight_layout()

def loss_idd(c):
    """
    This function plots loss data for Actor, Critic, and Temperature from the given CSV files 
    and saves the plot as a PNG image in the specified path.
    """
    plt.figure(figsize=(30, 12))

    # Plot Actor Loss
    actor_data = read_loss_data(path + f'/loss/dual_loss_{c}_pi.csv')
    plot_loss(actor_data, 'Actor', 'red', 1)

    # Plot Critic Loss
    critic_data = read_loss_data(path + f'/loss/dual_loss_{c}_q1.csv')
    plot_loss(critic_data, 'Critic', 'green', 2)

    # Plot Temperature Loss
    temperature_data = read_loss_data(path + f'/loss/dual_loss_{c}_temp.csv')
    plot_loss(temperature_data, 'Temperature', 'black', 3)

    # Save and close
    plt.savefig(path + f'/loss/dual_loss_visual/loss_{c}.png', format='png', dpi=300)
    plt.close('all')  # Properly close the plot to free up memory
    gc.collect()

if __name__ == "__main__":
    identifier = "example"  # Replace with the desired identifier.
    loss_idd(identifier)
