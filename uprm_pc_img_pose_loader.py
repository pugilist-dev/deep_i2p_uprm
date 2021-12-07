#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 12:11:01 2021

@author: raj
"""

import open3d
import torch.utils.data as data
import random
import numbers
import os
import os.path
import numpy as np
import struct
import math
import torch
import torchvision
import cv2
from PIL import Image
from torchvision import transforms
import bisect

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import options
from helper import farthest_sampler
import augmentation


def make_uprm_dataset(root_path, mode, opt):
    images = []
    point_clouds = []
    image_path = root_path + "interim/images/"
    pc_path = root_path + "interim/point_cloud/"
    if mode == "train":
        image_name = os.listdir(image_path+mode)
        pc_name = os.listdir(pc_path+mode)
        for name in image_name:
            if name.endswith(".png"):
                images.append(image_path+mode+"/"+name)
        for name in pc_name:
            if name.endswith(".ply"):
                point_clouds.append(pc_path+mode+"/"+name)
    if mode == 'val':
        if name.endswith(".png"):
                images.append(image_path+mode+"/"+name)
        for name in pc_name:
            if name.endswith(".ply"):
                point_clouds.append(pc_path+mode+"/"+name)
                
    return images, point_clouds
            

class uprm_loader(data.Dataset):
    def __init__(self, root, mode, opt: options.Options):
        super(uprm_loader, self).__init__()
        self.root = root
        self.opt = opt
        self.mode = mode
        
        # Farthest point sample
        self.farthest_sampler = farthest_sampler(dim = 3)
        self.images, self.pc = make_uprm_dataset(root, mode, opt)
    
    
    def augment_pc(self, pc_np):
        """
        Parameters
        ----------
        pc_np : 3xN, np.ndarray
            DESCRIPTION.
        Returns
        -------
        point cloud with added noise is returned
        """
        # add gaussian noise
        pc_np = augmentation.jitter_point_cloud(pc_np, sigma=0.01, clip=0.05)
        return pc_np
    
    
    def augment_img(self, img_np):
        """
        Parameters
        ----------
        img_np : HxWx3 image in np.ndarray
        Returns
        -------
        Augmented Image is returned 
        """
        # color perturbation
        #brightness = (0.8, 1.2)
        #contrast = (0.8, 1.2)
        #saturation = (0.8, 1.2)
        #hue = (-0.1, 0.1)
        #color_aug = transforms.ColorJitter.get_params(brightness, contrast, saturation, hue)
        my_transform = transforms.Compose([
            transforms.ColorJitter(0.7,0.5,0.5,0.5),
         ])
        img_color_aug_np = np.array(my_transform(Image.fromarray(img_np)))

        return img_color_aug_np
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        K = np.asarray([[964.828979, 0, 643.788025], [0, 964.828979, 484.407990], [0, 0, 1]], dtype=np.float32)
        
        

        