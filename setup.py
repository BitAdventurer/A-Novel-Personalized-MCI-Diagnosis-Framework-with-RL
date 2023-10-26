'''
We used cython to speed up the process.

Enter the command 'python setup.py build_ext --inplace' to cythonize and then run the code.
'''
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

module_names = [
    "train_network",
    "self_play_best",
    "Action",
    "Action_lap",
    "disconnection",
    "disconnection_laplace",
    "evaluate_best_player_val_p",
    "DualNetwork",
    "util",
    "gcn_util"
]

for module in module_names:
    setup(
        description=f"Cythonize {module}",
        ext_modules=cythonize(f"{module}.pyx"),
        include_dirs=[np.get_include()]
    )
