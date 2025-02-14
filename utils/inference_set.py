import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from torchvision import  datasets

from functions import *
from additional_feature import *

import json
import os
import logging
import copy
import cv2
import time
from random import randint
import glob
import random
from PIL import Image
from ois import optimal_system
from torchvision.models import mobilenet_v3_small
from torch.utils.data import RandomSampler
from tqdm.contrib import tzip
import torchvision.transforms as transforms
from torchvision.transforms import RandomHorizontalFlip, RandomRotation,RandomVerticalFlip, RandomCrop, CenterCrop, Normalize, GaussianBlur, RandomRotation

# devices
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'CHECK devices {device}')


# Dataset
class inference_set(Dataset):

    def __init__(self, images, mode=None, transform=None,  local_scaler=None, default_size=20,add_similarity=None):


        self.all_imgs = images
        #self.scaler = scaler
        self.transform = transform
        self.mode = mode
        self.default_size = default_size
        self.local_scaler = local_scaler
        self.add_sim = add_similarity
        
        
    def __len__(self):

        input_size = np.shape(self.all_imgs)[0] 
        return input_size

    def __getitem__(self, idx):
        
        
        trk_imgs = self.all_imgs[idx].copy() # copy to avoid data modification
        if self.add_sim is True:
            sim_imgs = get_similarity(trk_imgs)
            trk_imgs = np.concatenate([trk_imgs,sim_imgs], axis=0)
        trk_imgs = np.transpose(trk_imgs)
        
            
        #  ========== Transform images as tensor ========== #
        trk_imgs = transforms.ToTensor()(trk_imgs)
        
         
        if self.transform is not None:
 
            trk_imgs = self.transform(trk_imgs)
                    
                
        # Perfprm local scaling
        if self.local_scaler is not None:
            assert self.local_scaler in ['std_scl','norm_scl','l2norm_scl','bkg_std_scl']
            
            
            # Standard Scaler
            if self.local_scaler == 'std_scl':
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_standard_scaler(local_scaled_imgs)
            
            
            elif self.local_scaler == 'norm_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_norm_scaler(local_scaled_imgs)
                
            elif self.local_scaler == 'l2norm_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = tensor_L2Norm(local_scaled_imgs)
                
            elif self.local_scaler == 'bkg_std_scl' : 
                local_scaled_imgs = torch.clone(trk_imgs)
                local_scaled_imgs = sep_standard_scaler(local_scaled_imgs)
        
            
            assert torch.max(local_scaled_imgs)<5000, f'Values are not scaled {torch.max(local_scaled_imgs)}, {torch.min(local_scaled_imgs)}'    # No global scaler':
            
            
            imgs = local_scaled_imgs
            
        else:
            imgs = trk_imgs
            
        # Center Cropping
        imgs = CenterCrop(self.default_size)(imgs)
        
        #print(f'CHECK IMAGE Shape {imgs.size()}')
        # diff and p2p after scaling is performed
        if self.mode is None:
            pass
        
        
        else:
            if 'direct_diff' in self.mode:
                diff_imgs = get_diff_imgs(imgs)
                imgs = np.concatenate((imgs,diff_imgs), axis=0)
            
            if 'bramich_diff' in self.mode:
                diff_imgs = get_bramich_diff(imgs)
                imgs = np.concatenate((imgs,diff_imgs), axis=0)
            

            if 'p2p' in self.mode:
                p2p_imgs = get_p2p(imgs[:3]) # p2p done on raw only
                imgs = np.concatenate((imgs,p2p_imgs), axis=0)
                
        
            
        return imgs