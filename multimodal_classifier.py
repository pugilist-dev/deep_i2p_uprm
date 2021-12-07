#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 11:08:28 2021

@author: raj
"""

import torch
import torch.nn as nn
import torchvision
import numpy as np
import math
from collections import OrderedDict
import os
import random
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from options import Options

class MMClassifier():
    def __init__(self, opt: Options, writer):
        self.opt = opt
        self.writer = writer
        

class MMClassifierCoarse():
    def __init__(self, opt: Options, writer):
        self.opt = opt
        self.writer = writer