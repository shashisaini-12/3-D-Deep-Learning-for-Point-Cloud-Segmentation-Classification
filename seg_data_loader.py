# It is a dataloader for segmentation of S3DIS dataset for feeding the data in ContiNet architecture
''' Dataset for The aligned, reduced, partitioned S3DIS dataset 
    Provides functionality for train/test on partitioned sets as well 
    as testing on entire spaces via get_random_partitioned_space()
    '''

import os
import sys
from glob import glob
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class S3DIS(Dataset):
    def __init__(self, root, area_nums, split='train', npoints=4096, r_prob=0.25):
        self.root = root
        self.area_nums = area_nums # i.e. '1-4' # areas 1-4
        self.split = split.lower() # use 'test' in order to bypass augmentations
        self.npoints = npoints     # use  None to sample all the points
        self.r_prob = r_prob       # probability of rotation
        self.label_mapping = {
                                'ceiling'  : 0, 
                                'floor'    : 1, 
                                'wall'     : 2, 
                                'beam'     : 3, 
                                'column'   : 4, 
                                'window'   : 5,
                                'door'     : 6, 
                                'table'    : 7, 
                                'chair'    : 8, 
                                'sofa'     : 9, 
                                'bookcase' : 10, 
                                'board'    : 11,
                                'stairs'   : 12,
                                'clutter'  : 13
                            }

        # glob all Areas_ paths
        areas = glob(os.path.join(root, f'Area_[{area_nums}]*'))

        # check that datapaths are valid, if not raise error
        if len(areas) == 0:
            raise FileNotFoundError("NO VALID FILEPATHS FOUND!")

        for p in areas:
            if not os.path.exists(p):
                raise FileNotFoundError(f"PATH NOT VALID: {p} \n")

        # Store All Annotations directories which is inside the all Area_ directories    
        self.Annotation_directories_paths = [] #Store the all Annotation directories path
        for area in areas:
            self.Annotation_directories_paths += glob(os.path.join(area, '**\*Annotations'),
                                                       recursive=True)
            
        # It stores all .txt file path which is inside the Annotation folder for all Area_ directories
         #Stores all text files path in Annotation folder from all Area_(1-4)
        self.data_paths = []
        for path in self.Annotation_directories_paths:
            self.data_paths += glob(os.path.join(path, '**\*.txt'), recursive=True)
        
        # It stores all labels from the text files which path belongs to data_paths
        self.labels = []
        for fp in self.data_paths:
            label = fp.split('\\')[-1].split('_')[0]
            self.labels.append(label)
        # get unique space identifiers (area_##\\spacename_##_)
        # and this block of code used for testing
        self.space_ids = []
        for fp in self.data_paths:
            area, space = fp.split('\\')[-4:-2]
            space_id = '\\'.join([area, space]) 
            self.space_ids.append(space_id)


    def __getitem__(self, idx):
        # read data from txt file from pandas make it fast
        # just consider the first 3 cloumns and drop the rgb values
        space_data = pd.read_csv(self.data_paths[idx], delimiter=' ', usecols=(0, 1, 2), dtype=float, header=None)
        # It create new column named Labels and assign the corresponding labels mapping value.
        space_data['Labels'] = self.label_mapping[self.labels[idx]] #Insert new column Label
        space_data = space_data.values #convert pandas frame work to  array
        points = space_data[:, :3] # xyz points
        targets = space_data[:, 3] # Labels/Category

        # down sample point cloud
        if self.npoints:
            points, targets = self.downsample(points, targets)

        # add Gaussian noise to point set if not testing
        if self.split != 'test':
            # add N(0, 1/100) noise
            points += np.random.normal(0., 0.01, points.shape)

            # add random rotation to the point cloud with probability
            if np.random.uniform(0, 1) > 1 - self.r_prob:
                points = self.random_rotate(points)


        # Normalize Point Cloud to (0, 1)
        points = self.normalize_points(points)

        # convert numpy array from object type to float/int
        # points = points.astype(float)
        # targets = targets.astype(int)

        # convert to torch
        points = torch.from_numpy(points).type(torch.float32)
        targets = torch.from_numpy(targets).type(torch.LongTensor)

        return points, targets
        

    def get_random_partitioned_space(self):
        ''' Obtains a Random space. In this case the batchsize would be
            the number of partitons that the space was separated into.
            This is a special function for testing.
            '''

        # get random space id
        idx = random.randint(0, len(self.space_ids) - 1)
        space_id = self.space_ids[idx]

        # get all filepaths for randomly selected space
        space_paths = []
        space_labels = []
        for fpath in self.data_paths:
            if space_id in fpath:
                space_paths.append(fpath)
                label = fpath.split('\\')[-1].split('_')[0]
                space_labels.append(label)

        
        
        # assume npoints is very large if not passed
        if not self.npoints:
            self.npoints = 20000

        points = np.zeros((len(space_paths), self.npoints, 3))
        targets = np.zeros((len(space_paths), self.npoints))

        # # Type caste values inside the points and targets numpy array in float/int from string
        # points = points.astype(float)
        # targets = targets.astype(int)

        # obtain data
        for i, space_path in enumerate(space_paths):
            space_data = pd.read_csv(space_path, delimiter=' ', usecols=(0, 1, 2), dtype=float, header=None)
            space_data['Labels'] = self.label_mapping[space_labels[i]] # Add label column for assigning corrosponding labels
            space_data = space_data.values  #pandas frame turned into array
            _points = space_data[:, :3] # xyz points
            _targets = space_data[:, 3] # integer categories

            # downsample point cloud
            _points, _targets = self.downsample(_points, _targets)

            # add points and targets to batch arrays
            points[i] = _points
            targets[i] = _targets

        

        # convert to torch
        points = torch.from_numpy(points).type(torch.float32)
        targets = torch.from_numpy(targets).type(torch.LongTensor)

        return points, targets
        

    def downsample(self, points, targets):
        if len(points) > self.npoints:
            choice = np.random.choice(len(points), self.npoints, replace=False)
        else:
            # case when there are less points than the desired number
            choice = np.random.choice(len(points), self.npoints, replace=True)
        points = points[choice, :] 
        targets = targets[choice]

        return points, targets

    
    @staticmethod
    def random_rotate(points):
        ''' randomly rotates point cloud about vertical axis.
            Code is commented out to rotate about all axes
            '''
        # construct a randomly parameterized 3x3 rotation matrix
        phi = np.random.uniform(-np.pi, np.pi)
        theta = np.random.uniform(-np.pi, np.pi)
        psi = np.random.uniform(-np.pi, np.pi)

        rot_x = np.array([
            [1,              0,                 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi) ]])

        rot_y = np.array([
            [np.cos(theta),  0, np.sin(theta)],
            [0,                 1,                0],
            [-np.sin(theta), 0, np.cos(theta)]])

        rot_z = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi),  0],
            [0,              0,                 1]])

        # rot = np.matmul(rot_x, np.matmul(rot_y, rot_z))
        
        return np.matmul(points, rot_z)


    @staticmethod
    def normalize_points(points):
        points = points - points.min(axis=0)
        max_values = points.max(axis=0)
        
        # Create a mask to identify columns with non-zero max values
        non_zero_mask = max_values != 0
        
        # Normalize the columns with non-zero max values
        points[:, non_zero_mask] /= points[:, non_zero_mask].max(axis=0)
        
        return points

    def __len__(self):
        return len(self.data_paths)
       
            
