import pandas as pd
import matplotlib.pyplot as plt
import ast

import config
args = config.parser.parse_args()
path = args.path

def read_data(file_path):
    """
    Helper function to read and parse data from a given filepath.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = [ast.literal_eval(line.strip()) for line in file if line.strip()]
        return data
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return []
    except SyntaxError:
        print(f"Error: Syntax error in the data in the file {file_path}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def plot_metrics(metrics, labels, colors, linestyle, title, xlabel, ylabel, save_path):
    """
    Function to plot given metrics with specified attributes.
    """
    plt.figure(figsize=(20, 8))
    plt.title(title)
    
    for metric, label, color, line in zip(metrics, labels, colors, linestyle):
        plt.plot(metric, label=label, marker='o', linestyle=line, linewidth=1.5, markersize=1, color=color)

    plt.axhline(0, color='green', linestyle='-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, format='jpg', dpi=300)
    plt.close()

def plot_acc_val():
    """
    Function to plot different accuracy metrics such as accuracy, sensitivity, specificity, and F1.
    """
    metrics_files = [
        '/result/confusion_info/val_acc_plot_q.txt',
        '/result/confusion_info/val_sensitivity_plot_q.txt',
        '/result/confusion_info/val_specificy_plot_q.txt',
        '/result/confusion_info/val_f1_plot_q.txt'
    ]
    labels = ['Accuracy', 'Sensitivity', 'Specificity', 'F1 Score']
    colors = ['red', '#ff7f0e', 'blue', 'black']
    linestyles = ['-', '-', '-', '--']
    
    metrics = [read_data(path + file) for file in metrics_files]
    
    plot_metrics(
        metrics, labels, colors, linestyles,
        'Validation Accuracy Metrics', 'Iteration', 'Delta',
        path + '/result/val_acc.jpg'
    )

if __name__ == "__main__":
    plot_acc_val()
