#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fvanmaele
"""

import matplotlib.pyplot as plt
import json
from glob import glob

# %%
plt.rcParams['text.usetex'] = True

# %%
def generate_plots(data, num):   
    image = {}
    with open(data[0]) as f:
        image['fista_cd'] = json.load(f)
    with open(data[1]) as f:
        image['fista_mod'] = json.load(f)
    with open(data[2]) as f:
        image['fista'] = json.load(f)

    # iterations <-> ||xk - xkpp||
    k1 = image['fista']['k']
    k2 = image['fista_mod']['k']
    k3 = image['fista_cd']['k']
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

    #plt.xlim(500)
    ax1.set_yscale('log')
    #ax1.set_xlabel('iterations')
    ax1.set_ylabel(r'$\|x_k - x_{k-1}\|_F$')
    ax1.set_title('image{}'.format(num))
    ax1.plot(range(0, k1), image['fista']['solution_norm_diff'], label='FISTA')
    ax1.plot(range(0, k2), image['fista_mod']['solution_norm_diff'], label='FISTA-Mod')
    ax1.plot(range(0, k3), image['fista_cd']['solution_norm_diff'], label='FISTA-CD')

    # iterations to ||J(xk) - J(xkpp)||
    ax2.set_yscale('log')
    #ax2.set_xlabel('iterations')
    ax2.set_ylabel(r'$J(x_k) - J(x_{k-1})$')
    ax2.plot(range(0, k1), image['fista']['objective_norm_diff'], label='FISTA')
    ax2.plot(range(0, k2), image['fista_mod']['objective_norm_diff'], label='FISTA-Mod')
    ax2.plot(range(0, k3), image['fista_cd']['objective_norm_diff'], label='FISTA-CD')

    # iterations to PSNR
    ax3.set_yscale('linear')
    ax3.set_xlabel('iterations')
    ax3.set_ylabel('PSNR')
    ax3.plot(range(0, k1), image['fista']['psnr'], label='FISTA')
    ax3.plot(range(0, k2), image['fista_mod']['psnr'], label='FISTA-Mod')
    ax3.plot(range(0, k3), image['fista_cd']['psnr'], label='FISTA-CD')

    plt.legend()
    plt.savefig('image{}_analysis.png'.format(num), dpi=108)

# %%
data0 = glob('image0_*.json')
data0.sort()  # 0: fista_cd, 1: fista_mod, 2: fista
generate_plots(data0, 0)

data1 = glob('image1_*.json')
data1.sort()
generate_plots(data1, 1)

data2 = glob('image2_*.json')
data2.sort()
generate_plots(data2, 2)

data3 = glob('image3_*.json')
data3.sort()
generate_plots(data3, 3)

data4 = glob('image4_*.json')
data4.sort()
generate_plots(data4, 4)

data5 = glob('image5_*.json')
data5.sort()
generate_plots(data5, 5)

data6 = glob('image6_*.json')
data6.sort()
generate_plots(data6, 6)

data7 = glob('image7_*.json')
data7.sort()
generate_plots(data7, 7)