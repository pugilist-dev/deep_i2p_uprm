#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 16:38:23 2021

@author: raj
"""

import numpy as np
import os
import open3d
import cv2

class farthest_sampler:
    def __init__(self, dim = 3):
        self.dim = dim
    
    def calc_distance(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=0)
    
    def sample(self, pts, k):
        farthest_pts = np.zeros(self.dim, k)
        farthest_pts_idx = np.zeros(k, dtype=np.int)
        init_idx = np.random.randint(len(pts))
        farthest_pts[:, 0] = pts[:, init_idx]
        farthest_pts_idx[0] = init_idx
        distances = self.calc_distances(farthest_pts[:, 0:1], pts)
        for i in range(1, k):
            idx = np.argmax(distances)
            farthest_pts[:, i] = pts[:, idx]
            farthest_pts_idx[i] = idx
            distances = np.minimum(distances, self.calc_distances(farthest_pts[:, i:i+1], pts))
        return farthest_pts, farthest_pts_idx