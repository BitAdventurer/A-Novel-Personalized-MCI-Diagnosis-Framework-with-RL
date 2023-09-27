import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import util
import config
import numpy as np
import ast
import os
args = config.parser.parse_args()
path = args.path

import ast
import matplotlib.pyplot as plt
import config

# Grouping related functionalities into functions for modularity and readability
def read_list_from_file(file_path):
    with open(file_path, mode='rt', encoding='utf-8') as file:
        lines = [ast.literal_eval(line.strip()) for line in file.readlines()]
    return lines

def plot_graph(total_acc, sensitivity, specificity, f1, title, save_path):
    plt.title(title)
    plt.figure(figsize=(20, 8))
    plt.plot(total_acc, label='Acc', marker='o', ls='-', lw=1.5, markersize=1, color='red')
    plt.plot(sensitivity, label='Sen', marker='o', ls='-', lw=0.5, markersize=1, color='#ff7f0e')
    plt.plot(specificity, label='Spec', marker='o', ls='-', lw=0.5, markersize=1, color='blue')
    plt.plot(f1, label='F1', marker='o', ls='--', lw=0.5, markersize=1, color='black')
    plt.plot([0] * len(total_acc), ls='-', color='green')

    plt.xlabel('Iteration')
    plt.ylabel('Delta')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, format='jpg', dpi=300)
    plt.close()

def plot_acc_val():
    args = config.parser.parse_args()
    path = args.path

    # Building file paths dynamically using os.path.join
    total_acc = read_list_from_file(os.path.join(path, 'result', 'confusion_info', 'val_acc_plot_q_c.txt'))
    sensitivity = read_list_from_file(os.path.join(path, 'result', 'confusion_info', 'val_sensitivity_plot_q_c.txt'))
    specificity = read_list_from_file(os.path.join(path, 'result', 'confusion_info', 'val_specificy_plot_q_c.txt'))
    f1 = read_list_from_file(os.path.join(path, 'result', 'confusion_info', 'val_f1_plot_q_c.txt'))
    
    save_path = os.path.join(path, 'result', 'val_acc_c.jpg')
    plot_graph(total_acc, sensitivity, specificity, f1, 'Val', save_path)





