"""Cross-validation split visualization helpers."""

import numpy as np


def plot_cv_indices(cv, x, y, ax, n_splits, lw=10):
    """Visualize train/test membership for each split on a Matplotlib axis."""
    for split_idx, (train_idx, test_idx) in enumerate(cv.split(x, y)):
        indices = np.full(len(x), np.nan)
        indices[test_idx] = 1
        indices[train_idx] = 0

        ax.scatter(
            range(len(indices)),
            [split_idx + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap="coolwarm",
            vmin=-0.2,
            vmax=1.2,
        )

    ax.set(
        ylim=[n_splits + 0.2, -0.2],
        xlim=[0, len(x)],
        xlabel="Sample index",
        ylabel="CV split",
    )
    ax.set_title("Cross-validation splits", fontsize=14)
