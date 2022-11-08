#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ferdinand Vanmaele
"""

import json
import re
import matplotlib.pyplot as plt

from glob import glob
from cycler import cycler

# %%
def show_plots(files, label):
    plt.clf()
    plt.yscale('log')
    plt.rc('lines', linewidth=1.5)
    
    default_cycler = (cycler(color=['r', 'g', 'b', 'y']) *
                      cycler(linestyle=['-', '--']))
    plt.rc('axes', prop_cycle=default_cycler)
    
    for filename in files:
        # Retrieve algorithm from file name
        algo = re.match("^\w_\w+_\w+_\d+_trials_fre_(\w+)", filename)
        algo = algo.group(1)
        
        # Process data
        data = {}
        with open(filename, 'r') as f:
            data = json.load(f)    
        plt.plot(range(1, len(data[label])+1), data[label], label=algo)
        plt.legend()
    
    plt.show()


# %%
A_json = glob("A_*.json")  # Gaussian matrix
A_json.sort()
F_json = glob("F_*.json")  # Partial Fourier matrix
F_json.sort()

# %%
show_plots(A_json, 'error')

# %%
show_plots(F_json, 'error')

# %%
show_plots(A_json, 'cputime')

# %%
show_plots(F_json, 'cputime')