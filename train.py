#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 10:14:14 2021

@author: raj
"""

import open3d as o3d
import time
import copy
import numpy as np
import math
import os
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use('Agg')

import options
from multimodal_classifier import MMClassifier, MMClassifierCoarse
from uprm_pc_img_pose_loader import uprm_loader

if __name__ == '__main__':
    opt = options.Options()
    logdir = './runs/'+str(opt.version)
    if os.path.isdir(logdir):
        user_answer = input("The log directory %s exists, do you want to delete it? (y or n) : " % logdir)
        if user_answer == 'y':
            # delete log folder
            shutil.rmtree(logdir)
        else:
            exit()
    else:
        os.makedirs(logdir)
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(log_dir=logdir)
    
    trainset = uprm_loader(opt.dataroot, 'train', opt)