#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ferdinand Vanmaele
"""

from glob import glob
import re
import shutil
import os
import numpy as np
import cv2


# %%
images = glob('data_all/*')
images.sort()

cropped = 'data_cropped/'
os.mkdir(cropped)

# %% Detect faces in image and crop to region (haar cascade)
for f in images:
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    
    face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for i, (x, y, w, h) in enumerate(faces):
        #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faces = img[y:y + h, x:x + w]

        cv2.imwrite(cropped + os.path.splitext(os.path.basename(f))[0] 
                    + '_face{}'.format(i) 
                    + '.jpg', faces)

# %% Remove outliers in size
cropped = glob('data_cropped/*')

for f in cropped:
    img = cv2.imread(f)
    h, w, _ = img.shape
    
    if h < 100 or h > 130 or w < 100 or w > 130:
        print("Removing: " + f)
        os.remove(f)
        next

# %% Resize images to maximum height/width (130)
cropped = glob('data_cropped/*')

for f in cropped:
    img = cv2.imread(f)
    h, w, _ = img.shape
    assert(h == w)

    output = cv2.resize(img, (130, 130))
    cv2.imwrite(f, output)

# %% Count occurences of each person
people = {}

for idx, image in enumerate(cropped):
    v = re.split(r'([\w-]+_)+(\d{4})', image)
    person   = v[1].removesuffix('_')
    instance = v[2]

    try:
        people[person].append(idx)
    except KeyError:
        people[person] = [idx]

        
# %% Only consider persons with a certain range of samples
min_samples = 5
max_samples = 10
num_train = 4
num_verif = 1

training_set = {}
verification_set = {}

for person in people:
    idx = people[person]

    if len(idx) >= min_samples and len(idx) <= max_samples:
        idx_train = np.random.choice(idx, num_train)
        idx_verif = np.random.choice(np.setdiff1d(idx, idx_train), num_verif)

        training_set[person] = idx_train
        verification_set[person] = idx_verif


# %%
target = 'data_training/'
os.mkdir(target)

for person in training_set:
    for id in training_set[person]:
        shutil.copy(cropped[id], target)

# %%
target = 'data_verification/'
os.mkdir(target)

for person in verification_set:
    for id in verification_set[person]:
        shutil.copy(cropped[id], target)