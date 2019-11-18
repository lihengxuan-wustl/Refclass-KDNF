# Refclass-KDNF
This is the repo for the large reference class k-dnf experiment for AAAI 2020 paper "More accurate learning of k-DNF reference classes"
Experiment setup is based on: https://github.com/marcotcr/anchor-experiments

How to install packages:
pip install lime==0.1.1.29 sklearn matplotlib seaborn numpy argparse xgboost==0.4a30 cython

How to compile Cython:
python setup.py

Before running:
Please make sure output folder "tmp" and "out_pickles" are present as empty folders are not checked in by git.

How to run:
compute_explanations.py is the main entry point that you can run with args. However the various files starting with "run" may help you automate the process on your system.

How to collect results:
All results go to "out_pickles" folder with format
Line 1: expression of kdnf
Line 2: Precision
Line 3: coverage
Files with name agg_... may help you automate the process
All running logs go to "tmp" folder
