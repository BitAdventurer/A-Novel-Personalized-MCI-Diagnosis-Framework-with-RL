import pandas as pd
import matplotlib.pyplot as plt
import config
import gc
from datetime import datetime

args = config.parser.parse_args()
path = args.path


def loss_idd(c):
    """
    This function plots loss data for Actor, Critic, and Temperature from the given CSV files 
    and saves the plot as a PNG image in the specified path.
    
    :param c: Identifier to distinguish between different loss data files and save names.
    """

    # Setup Figure and Subplots
    plt.figure(figsize=(30, 12))

    # Plot Actor Loss
    plt.subplot(1, 3, 1)
    actor_data = pd.read_csv(path + f'/loss/dual_loss_{c}_pi.csv')
    plt.title('Actor')
    plt.plot(actor_data['0'], label='Actor', marker='o', lw=2, ls='-', markersize=2, color='red')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Plot Critic Loss
    plt.subplot(1, 3, 2)
    critic_data = pd.read_csv(path + f'/loss/dual_loss_{c}_q1.csv')
    plt.title('Critic')
    plt.plot(critic_data['0'], label='Critic', marker='o', lw=2, ls='-', markersize=2, color='green')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Plot Temperature Loss
    plt.subplot(1, 3, 3)
    temperature_data = pd.read_csv(path + f'/loss/dual_loss_{c}_temp.csv')
    plt.title('Temperature')
    plt.plot(temperature_data['0'], label='Temperature', marker='o', lw=2, ls='-', markersize=2, color='black')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the Plot as PNG
    plt.savefig(path + f'/loss/dual_loss_visual/loss_{c}.png', format='png', dpi=300)
    
    # Close the plots and free up memory
    plt.close('all')
    gc.collect()


if __name__ == "__main__":
    identifier = "example"  # Replace with the desired identifier.
    loss_idd(identifier)
