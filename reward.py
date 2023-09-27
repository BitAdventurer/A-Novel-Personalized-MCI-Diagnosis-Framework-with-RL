import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import util
import config
import numpy as np
import ast
import wandb
args = config.parser.parse_args()
path = args.path

def plot_acc_val():
    """
    Function to plot different accuracy metrics such as accuracy, sensitivity, specificity, and F1.
    """
    
    # Read and process accuracy data
    with open(path + '/result/confusion_info/val_acc_plot_q.txt', 'r', encoding='utf-8') as f:
        total_acc_q = [ast.literal_eval(line) for line in f.readlines()]

    # Read and process sensitivity data
    with open(path + '/result/confusion_info/val_sensitivity_plot_q.txt', 'r', encoding='utf-8') as f:
        sensitivity_q = [ast.literal_eval(line) for line in f.readlines()]

    # Read and process specificity data
    with open(path + '/result/confusion_info/val_specificy_plot_q.txt', 'r', encoding='utf-8') as f:
        specificity_q = [ast.literal_eval(line) for line in f.readlines()]

    # Read and process F1 data
    with open(path + '/result/confusion_info/val_f1_plot_q.txt', 'r', encoding='utf-8') as f:
        f1_q = [ast.literal_eval(line) for line in f.readlines()]
    
    # Plotting
    plt.figure(figsize=(20, 8))
    plt.title('Validation Accuracy Metrics')
    plt.plot(total_acc_q, label='Accuracy', marker='o', linestyle='-', linewidth=1.5, markersize=1, color='red')
    plt.plot(sensitivity_q, label='Sensitivity', marker='o', linestyle='-', linewidth=0.5, markersize=1, color='#ff7f0e')
    plt.plot(specificity_q, label='Specificity', marker='o', linestyle='-', linewidth=0.5, markersize=1, color='blue')
    plt.plot(f1_q, label='F1 Score', marker='o', linestyle='--', linewidth=0.5, markersize=1, color='black')
    plt.axhline(0, color='green', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Delta')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(path + '/result/val_acc.jpg', format='jpg', dpi=300)
    plt.close()