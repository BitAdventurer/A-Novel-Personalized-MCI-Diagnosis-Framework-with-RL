"""Build Cython extensions for the MCI-RL training pipeline.

Run:
    python setup.py build_ext --inplace
"""

from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup


CYTHON_MODULES = [
    "train_network",
    "self_play_best",
    "Action",
    "Action_lap",
    "disconnection",
    "disconnection_laplace",
    "evaluate_best_player_val_p",
    "DualNetwork",
    "util",
    "gcn_util",
]


def make_extensions():
    """Create extension definitions for packaged .pyx modules."""
    extensions = []
    for module_name in CYTHON_MODULES:
        source = Path("src") / "mci_rl" / f"{module_name}.pyx"
        if source.exists():
            extensions.append(
                Extension(
                    f"mci_rl.{module_name}",
                    [str(source)],
                    include_dirs=[np.get_include()],
                )
            )
    return extensions


setup(
    name="mci-rl-sparse-graph",
    description="Cython extensions for sparse graph RL-based MCI diagnosis",
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=cythonize(make_extensions(), language_level=3),
)
