#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ferdinand Vanmaele
"""

from glob import glob
import re
import shutil
import os

# %% Count occurences of each person
images = glob('data_all/*')
images.sort()
people = {}

for idx, image in enumerate(images):
    v = re.split(r'([\w-]+_)+(\d{4})', image)
    person   = v[1].removesuffix('_')
    instance = v[2]

    try:
        people[person].append(idx)
    except KeyError:
        people[person] = [idx]
        
# %% Remove images with less than X occurences
min_samples = 5
max_samples = 20
filtered = {}

for person in people:
    idx = people[person]

    if len(idx) >= min_samples and len(idx) <= max_samples:
        filtered[person] = idx
        
# %% Write out the results
target = 'data/'
try:
    os.mkdir(target)
except FileExistsError:
    pass

for person in filtered:
    for id in filtered[person]:
        shutil.copy(images[id], target)