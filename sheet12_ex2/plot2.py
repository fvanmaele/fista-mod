#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 22:06:24 2022

@author: fvanmaele
"""

import json
import matplotlib.pyplot as plt
from scipy.io import mmread

# %%
data = {}
with open('fista_mod_sigma1.0e-03_tol1.0e-08_img0.json') as f:
    data = json.load(f)
    
# %%
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.set_yscale('log')
ax1.plot(range(0, data['k']), data['solution_norm_diff'])
ax1.set_ylabel('sol_norm_diff')
ax2.set_yscale('log')
ax2.set_ylabel('obj_norm_diff')
ax2.plot(range(0, data['k']), data['objective_norm_diff'])
fig.savefig('fista_mod_uncropped_norm_diff.jpg', dpi=108)

# %%
plt.yscale('linear')
plt.plot(range(0, len(data['solution'])), data['solution'])
plt.savefig('fista_mod_uncropped.jpg', dpi=108)

# %%
B = mmread('B_dsize508_imsize62500.mtx')

# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

plt.imshow((B@data['solution']).reshape(250, 250), cmap='gray')
plt.savefig('fista_mod_recovered.jpg', dpi=108)

# %%