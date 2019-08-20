import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from file_io import Events, read_2d_spikes


#########################################################
# Utility functions
#########################################################
def _train_test_split_classification(datasets, labels=None, train_size=0.8, seed=42):
    if not isinstance(datasets, (list, tuple)):
        datasets = [datasets]
        labels = [labels]
    train = []
    test = []
    lbls = ["sample", "label"]

    # Split each data subset into desired train test
    for idx, dset in enumerate(datasets):
        if labels[idx] is not None:
            x_trn, x_tst, y_trn, y_tst = train_test_split(dset, labels[idx], train_size=train_size, random_state=seed)
            trn = pd.DataFrame({lbls[0]: x_trn, lbls[1]: y_trn})
            tst = pd.DataFrame({lbls[0]: x_tst, lbls[1]: y_tst})
        else:
            trn, tst = train_test_split(dset, train_size=train_size, random_state=seed)
            if not isinstance(trn, pd.DataFrame):
                trn = pd.DataFrame(trn, columns=lbls)
                tst = pd.DataFrame(tst, columns=lbls)
            else:
                trn.columns = lbls
                tst.columns = lbls
                trn.reset_index(inplace=True, drop=True)
                tst.reset_index(inplace=True, drop=True)

        train.append(trn)
        test.append(tst)

    train = pd.concat(train)
    test = pd.concat(test)

    return train, test


#########################################################
# Neurmorphic Caltech 101
#########################################################
class NCaltechDataset(Dataset):
    def __init__(self, root_dir, im_transform=None, lbl_transform=None):
        # Make list of file names and labels
        train_names = []
        train_labels = []
        for label, obj in enumerate(os.listdir(root_dir)):
            obj_names = os.listdir(os.path.join(root_dir, obj))
            obj_names = [os.path.join(obj, obj_name) for obj_name in obj_names]
            train_names += obj_names
            
            obj_labels = np.ones(len(obj_names), dtype=int) * label
            train_labels += obj_labels.tolist()
        
        # Make dataframe containing info
        self.im_info = pd.DataFrame({
            'name': train_names,
            'labels': train_labels
        })
        self.root_dir = root_dir
        self.im_transform = im_transform
        self.lbl_transform = lbl_transform
        
    def __len__(self):
        return len(self.im_info)
    
    def __getitem__(self, idx):
        pnt_name = os.path.join(self.root_dir, self.im_info.iloc[idx, 0])
        with open(pnt_name, 'rb') as f:
            sample = pickle.load(f)
        label = self.im_info.iloc[idx, 1]
        
        # Apply transforms
        if self.im_transform:
            sample = self.im_transform(sample)
        if self.lbl_transform:
            label = self.lbl_transform(label)
            
        return (sample, label)


#########################################################
# Neuromorphic MNIST
#########################################################
class NMNISTDataset(Dataset):
    def __init__(self, dataset_path, sample_file, sampling_time, sample_length):
        self.path = dataset_path 
        self.samples = np.loadtxt(sample_file).astype('int')
        self.sampling_time = sampling_time
        self.n_time_bins = int(sample_length / sampling_time)

    def __getitem__(self, index):
        input_index  = self.samples[index, 0]
        class_label  = self.samples[index, 1]
        
        input_spikes = read_2d_spikes(self.path + str(input_index.item()) + '.bs2')
        input_spikes = input_spikes.to_spike_tensor(torch.zeros((2,34,34,self.n_time_bins)), 
                                                    sampling_time=self.sampling_time)
        desired_class = torch.zeros((10, 1, 1, 1))
        desired_class[class_label,...] = 1
        return input_spikes, desired_class, class_label
    
    def __len__(self):
        return self.samples.shape[0]


############################
# N-Cars
############################
class NCarsDataset(Dataset):
    def __init__(self, root_dir, sampling_time, sample_length):
        # Construct dataframe with image names and labels
        car_names = os.listdir(os.path.join(root_dir, 'cars'))
        car_names = [os.path.join('cars', car) for car in car_names]
        car_labels = np.ones(len(car_names), dtype=int)
        background_names = os.listdir(os.path.join(root_dir, 'background'))
        background_names = [os.path.join('background', backgr) for backgr in background_names]
        background_labels = np.zeros(len(background_names), dtype=int)
        names = car_names + background_names
        labels = car_labels.tolist() + background_labels.tolist()
        
        # Assign values to class
        self.im_info = pd.DataFrame({
            'name':  names,
            'label': labels})
        self.root_dir = root_dir
        self.im_transform = im_transform
        self.lbl_transform = lbl_transform
        self.sampling_time = sampling_time
        self.n_time_bins = int(sample_length / sampling_time)
        
    def __len__(self):
        return len(self.im_info)
    
    def __getitem__(self, idx):
        # Load image & label
        im_name = os.path.join(self.root_dir, self.im_info.iloc[idx, 0])
        image = read_2d_spikes(im_name)
        if len(image.x) > len(image.y):
            image.x = image.x[:len(image.y)]
        image = image.to_spike_tensor(torch.zeros(2,100,120,self.n_time_bins),
                                    sampling_time=self.sampling_time)

        label = self.im_info.iloc[idx, 1] 
            
        sample = {'image': image, 'label': label}
        return sample


if __name__ == "__main__":
    # a = ["a", "b", "c", "d", "e"]
    a = np.random.rand(10)
    b = np.random.rand(10)
    data = {"a": a, "b": b}
    df = pd.DataFrame(data)
    print(df)
    print("\n")

    train, test = _train_test_split_classification(a, b)
    print(train)
    print(test)
    print("\n")

    train, test = _train_test_split_classification(df)
    print(train)
    print(test)
    