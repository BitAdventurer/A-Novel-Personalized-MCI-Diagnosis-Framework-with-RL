'''
We used cython to speed up the process.

Enter the command 'python setup.py build_ext -inplace' to cythonize and then run the code.
'''
from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy as np
#ext = Extension(name="hello", sources=["hello.pyx"]) 
#setup(ext_modules=cythonize(ext))

setup(description="C",ext_modules=cythonize("train_network.pyx"), include_dirs=[np.get_include()])
setup(description="C",ext_modules=cythonize("self_play_best.pyx"), include_dirs=[np.get_include()])
setup(description="C",ext_modules=cythonize("Action.pyx"), include_dirs=[np.get_include()])
setup(description="C",ext_modules=cythonize("Action_lap.pyx"), include_dirs=[np.get_include()])
setup(description="C",ext_modules=cythonize("disconnection.pyx"), include_dirs=[np.get_include()])
setup(description="C",ext_modules=cythonize("disconnection_laplace.pyx"), include_dirs=[np.get_include()])
setup(description="C",ext_modules=cythonize("evaluate_best_player_test_p.pyx"), include_dirs=[np.get_include()])
setup(description="C",ext_modules=cythonize("evaluate_best_player_test_c.pyx"), include_dirs=[np.get_include()])
setup(description="C",ext_modules=cythonize("evaluate_best_player_val_p.pyx"), include_dirs=[np.get_include()])
setup(description="C",ext_modules=cythonize("evaluate_best_player_val_c.pyx"), include_dirs=[np.get_include()])
setup(description="C",ext_modules=cythonize("DualNetwork.pyx"), include_dirs=[np.get_include()])
setup(description="C",ext_modules=cythonize("util.pyx"), include_dirs=[np.get_include()])
setup(description="C",ext_modules=cythonize("gcn_util.pyx"), include_dirs=[np.get_include()])


