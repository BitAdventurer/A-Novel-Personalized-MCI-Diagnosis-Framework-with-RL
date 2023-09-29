import pandas as pd
import matplotlib.pyplot as plt
import config

args = config.parser.parse_args()
path = args.path


def loss_idd(i):
    data = pd.read_csv(f'{path}/baseline/fold{args.fold}/baseline_loss{i}.csv')

    plt.plot(data['0'], label='train', marker='o', ls='-', markersize=2)
    plt.plot(data['0.1'], label='val', marker='o', ls='-', markersize=2)
    plt.plot(data['0.2'], label='test', marker='o', ls='-', markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{path}/baseline/fold{args.fold}/loss.png', format='png', dpi=300)
    plt.show()  # Moved here to actually show the plot


if __name__ == "__main__":
    # For demonstration purposes, assuming `i` is 1. Replace with appropriate value or loop over multiple values.
    loss_idd(1)
