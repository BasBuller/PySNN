import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from .file_io import Events, read_2d_spikes


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


def _list_dir_content(root_dir):
    content = {}
    for root, dirs, files in os.walk(root_dir):
        for label, name in enumerate(dirs):
            subdir = os.path.join(root, name)
            dir_content = os.listdir(subdir) 
            content[name] = [os.path.join(subdir, im) for im in dir_content]
    return content


def _concat_dir_content(content):
    ims = []
    labels = []
    names = []
    for idx, (name, data) in enumerate(content.items()):
        if not isinstance(data, (list, tuple)):
            data = [data]
        ims += data
        labels += [idx for _ in range(len(data))]
        names += [name for _ in range(len(data))]
    df = pd.DataFrame({"sample": ims, "label": labels})
    return df, names


def train_test(root_dir, train_size=0.8, seed=42):
    content = _list_dir_content(root_dir)
    data, _ = _concat_dir_content(content)
    train, test = _train_test_split_classification(data, train_size=train_size, seed=seed)
    return train, test


class NeuromorphicDataset(Dataset):
    def __init__(self, data, sampling_time, sample_length, height, width, im_transform=None, lbl_transform=None):
        self.data = data
        self.im_transform = im_transform
        self.lbl_transform = lbl_transform

        self.sampling_time = sampling_time
        self.sample_length = sample_length
        self.height = height
        self.width = width
        self.n_time_bins = int(sample_length / sampling_time)
        self.im_template = torch.zeros((2, height, width, self.n_time_bins))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Image sample
        im_name = self.data.iloc[idx, 0]
        input_spikes = read_2d_spikes(im_name)
        sample = input_spikes.to_spike_tensor(self.im_template, sampling_time=self.sampling_time)

        # Label
        label = self.data.iloc[idx, 1]
        
        # Apply transforms
        if self.im_transform:
            sample = self.im_transform(sample)
        if self.lbl_transform:
            label = self.lbl_transform(label)
            
        return (sample, label)


#########################################################
# Neurmorphic Caltech 101
#########################################################
def ncaltech_train_test(root="/home/basbuller/stack/thesis/code/datasets/n-caltech101/Caltech101/", 
                        sampling_time=1, 
                        sample_length=300, 
                        height=200, 
                        width=300, 
                        im_transform=None, 
                        lbl_transform=None):
    train, test = train_test(root)
    train_dataset = NeuromorphicDataset(train, sampling_time, sample_length, height, width, 
                                        im_transform=im_transform, lbl_transform=lbl_transform)
    test_dataset = NeuromorphicDataset(test, sampling_time, sample_length, height, width, 
                                        im_transform=im_transform, lbl_transform=lbl_transform)
    return train_dataset, test_dataset


#########################################################
# Neuromorphic MNIST
#########################################################
def nmnist_train_test(root="/home/basbuller/stack/thesis/code/datasets/n-mnist/",
                      sampling_time=1, 
                      sample_length=300, 
                      height=34, 
                      width=34, 
                      im_transform=None, 
                      lbl_transform=None):
    train_content = _list_dir_content(root + "Train")
    train, _ = _concat_dir_content(train_content)
    train_dataset = NeuromorphicDataset(train, sampling_time, sample_length, height, width, 
                                        im_transform=im_transform, lbl_transform=lbl_transform)

    test_content = _list_dir_content(root + "Test")
    test, _ = _concat_dir_content(test_content)
    test_dataset = NeuromorphicDataset(test, sampling_time, sample_length, height, width, 
                                        im_transform=im_transform, lbl_transform=lbl_transform)

    return train_dataset, test_dataset


#########################################################
# Neuromorphic MNIST
#########################################################
def ncars_train_test(root="/home/basbuller/stack/thesis/code/datasets/n-cars/",
                     sampling_time=1, 
                     sample_length=100, 
                     height=100, 
                     width=120, 
                     im_transform=None, 
                     lbl_transform=None):
    train_content = _list_dir_content(root + "train")
    train, _ = _concat_dir_content(train_content)
    train_dataset = NeuromorphicDataset(train, sampling_time, sample_length, height, width, 
                                        im_transform=im_transform, lbl_transform=lbl_transform)

    test_content = _list_dir_content(root + "test")
    test, _ = _concat_dir_content(test_content)
    test_dataset = NeuromorphicDataset(test, sampling_time, sample_length, height, width, 
                                        im_transform=im_transform, lbl_transform=lbl_transform)

    return train_dataset, test_dataset


if __name__ == "__main__":
    ncalt_train, ncalt_test = ncaltech_train_test()
    nmnist_train, nmnist_test = nmnist_train_test()
    ncars_train, ncars_test = ncars_train_test()

    print("caltech")
    print(ncalt_train[0][0].shape)
    print(ncalt_train[0][1])
    print(len(ncalt_train))

    print("\nnmnist")
    print(nmnist_train[0][0].shape)
    print(nmnist_train[0][1])
    print(len(nmnist_train))

    print("\nncars")
    print(ncars_train[0][0].shape)
    print(ncars_train[0][1])
    print(len(ncars_train))


############################
# Boolean
############################
class _Boolean(Dataset):
    def __init__(self, data_encoder=None, data_transform=None, noise_transform=None, lbl_transform=None, repeats=1):
        self.data_encoder = data_encoder
        self.data_transform = data_transform
        self.lbl_transform = lbl_transform
        self.noise_transform = noise_transform
        self.data = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float)
        self.data = torch.repeat_interleave(self.data, int(repeats), dim=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        # Sample transforms
        if self.noise_transform:
            sample = self.noise_transform(sample)
        if self.data_transform:
            sample = self.data_transform(sample)
        if self.data_encoder:
            sample = self.data_encoder(sample)

        # Label transforms
        if self.lbl_transform:
            label = self.lbl_transform(label)

        return sample, label


class XOR(_Boolean):
    def __init__(self, data_encoder=None, data_transform=None, noise_transform=None, lbl_transform=None, repeats=1):
        super(XOR, self).__init__(data_encoder, data_transform, noise_transform, lbl_transform, repeats)
        self.labels = torch.tensor([[0], [1], [1], [0]])


class AND(_Boolean):
    def __init__(self, data_encoder=None, data_transform=None, noise_transform=None, lbl_transform=None, repeats=1):
        super(AND, self).__init__(data_encoder, data_transform, noise_transform, lbl_transform, repeats)
        self.labels = torch.tensor([[0], [0], [0], [1]])


class OR(_Boolean):
    def __init__(self, data_encoder=None, data_transform=None, noise_transform=None, lbl_transform=None, repeats=1):
        super(OR, self).__init__(data_encoder, data_transform, noise_transform, lbl_transform, repeats)
        self.labels = torch.tensor([[0], [1], [1], [1]])


############################
# Transforms
############################
class DiscretizeFloat():
    def __call__(self, tensor):
        tensor = tensor.int()
        return tensor.float()


class BooleanNoisy():
    def __init__(self, low_thresh, high_thresh):
        self.low_distr = torch.distributions.uniform.Uniform(0., low_thresh) 
        self.high_distr = torch.distributions.uniform.Uniform(high_thresh, 1.)

    def __call__(self, x):
        for idx, sample in enumerate(x):
            if sample == 0:
                x[idx] = self.low_distr.sample()
            elif sample == 1:
                x[idx] = self.high_distr.sample()
        return x
