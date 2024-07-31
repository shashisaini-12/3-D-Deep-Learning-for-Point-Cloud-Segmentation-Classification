# It is a dataloader for Classification of ModelNet40(Point Cloud data not a mesh) Dataset, it will augment the data also to mitigate the effect of class imbalance.
## This dataloader do data augmentation to the minority classes (minority classes are 
## those which count is less than 150)

import random
import os
import sys
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class Modelnet40Dataset(Dataset):
    def __init__(self, root, npoints=4096, split='train', class_choice=None, 
                 normalize=True, data_augmentation=True):
        self.root = root
        self.split = split.lower()
        self.npoints = npoints
        self.data_augmentation = data_augmentation
        self.special_paths = []
        self.label_mapping = {
            'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4, 'bottle': 5, 'bowl': 6, 'car': 7,
            'chair': 8, 'cone': 9, 'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13, 'dresser': 14, 'flower_pot': 15,
            'glass_box': 16, 'guitar': 17, 'keyboard': 18, 'lamp': 19, 'laptop': 20, 'mantel': 21, 'monitor': 22,
            'night_stand': 23, 'person': 24, 'piano': 25, 'plant': 26, 'radio': 27, 'range_hood': 28, 'sink': 29,
            'sofa': 30, 'stairs': 31, 'stool': 32, 'table': 33, 'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37,
            'wardrobe': 38, 'xbox': 39
        }

        class_list = os.listdir(self.root)

        # Test and train paths
        test_train_paths = []
        for i in class_list:
            test_train_paths += glob(os.path.join(self.root, i))

        all_train_paths = []
        all_test_paths = []
        for paths in test_train_paths:
            all_train_paths += glob(os.path.join(paths, '**\*train'), recursive=True)
            all_test_paths += glob(os.path.join(paths, '**\*test'), recursive=True)

        # Final all train file paths and labels
        self.final_train_paths = []
        for i in all_train_paths:
            self.final_train_paths += glob(os.path.join(i, '**\*.npy'), recursive=True)

        self.train_labels = []
        for i in self.final_train_paths:
            label = i.split('\\')[-3]
            label = self.label_mapping[label]
            self.train_labels.append(label)

        # Final all test file paths and test labels
        self.final_test_paths = []
        for j in all_test_paths:
            self.final_test_paths += glob(os.path.join(j, '**\*.npy'), recursive=True)

        self.test_labels = []
        for i in self.final_test_paths:
            label = i.split('\\')[-3]
            label = self.label_mapping[label]
            self.test_labels.append(label)

        # Create validation data from the train data
        self.final_valid_paths = []
        self.valid_labels = []
        for i in range(0, 500):
            index = random.randint(0, len(self.final_train_paths))
            self.final_valid_paths.append(self.final_train_paths.pop(index))
            self.valid_labels.append(self.train_labels.pop(index))

        # Augment the classes with fewer than 150 samples
        if data_augmentation:
            self.augment_minority_classes()

        if self.split == 'train':
            self.data_paths = self.final_train_paths
            self.labels = self.train_labels
        elif self.split == 'valid':
            self.data_paths = self.final_valid_paths
            self.labels = self.valid_labels
        elif self.split == 'test':
            self.data_paths = self.final_test_paths
            self.labels = self.test_labels

    def __getitem__(self, idx):
        points = np.load(self.data_paths[idx])
        targets = self.labels[idx]
        
        if self.npoints:
            points = self.downsample(points)


        # Normalize Point Cloud to (0, 1)
        points = self.normalize_points(points)

        # converting points in torch tensor
        points = torch.from_numpy(points).type(torch.float32)
        targets = torch.tensor(targets)

        return points, targets

    def augment_minority_classes(self, min_samples=150):
        # Count the number of samples per class
        class_counts = {}
        for label in self.train_labels:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1

        # Augment the classes with fewer than min_samples
        for label, count in class_counts.items():
            if count < min_samples:
                sample_indices = [i for i, l in enumerate(self.train_labels) if l == label]
                for _ in range(min_samples - count):
                    # Randomly select a sample from the class
                    sample_idx = random.choice(sample_indices)
                    points = np.load(self.final_train_paths[sample_idx])

                    # Apply a combination of transformations
                    points = self.random_scale(points)
                    points = self.random_rotate(points)
                    points = self.add_jitter(points)

                    # Add the augmented sample to the dataset
                    self.final_train_paths.append(self.final_train_paths[sample_idx])
                    self.train_labels.append(label)

    def downsample(self, points):
        if len(points) > self.npoints:
            choice = np.random.choice(len(points), self.npoints, replace=False)
        else:
            # case when there are less points than the desired number
            choice = np.random.choice(len(points), self.npoints, replace=True)
        points = points[choice, :]
         
        return points



    @staticmethod
    def random_scale(points, scale_range=(0.8, 1.2)):
        scale_factor = np.random.uniform(*scale_range)
        return points * scale_factor

    @staticmethod
    def random_rotate(points):
        angle = np.random.uniform(-np.pi, np.pi)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])                                                                                                       
        return np.matmul(points, rotation_matrix.T)

    @staticmethod
    def add_jitter(points, sigma=0.01, clip=0.05):
        jittered_points = points + np.clip(sigma * np.random.randn(*points.shape), -1 * clip, clip)
        return jittered_points

    #@staticmethod
    #def normalize_points(points):
       # points = points - points.min(axis=0)
       # max_values = points.max(axis=0)
        #if np.all(max_values != 0): 
           # points /= points.max(axis=0)
        #else:
            # Handle the case where some columns are all zeros
           # pass
        #return points
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
