import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from .spike_file_io import read2Dspikes


############################
# Sine
############################
class SineData(Dataset):
    """Generate simple sine dataset containing multiple samples of a single sine wave."""
    def __init__(self, 
                 A=1, 
                 M=5, 
                 n_samples=3,
                 phase_range=(0,1),
                 t_range=(0,1),
                 encoder=None,
                 label_encoder=None,
                 data_transform=None,
                 label_transform=None):
        self.A = A
        self.M = M
        self.n_samples = n_samples
        self.encoder = encoder
        self.label_encoder = label_encoder
        self.data_transform = data_transform
        self.label_transform = label_transform
        
        # Construct sines & labels
        self.sines = []
        self.labels = []
        t = np.linspace(t_range[0], t_range[1], M)
        phases = np.linspace(phase_range[0], phase_range[1], self.n_samples)
        for idx, phase in enumerate(phases):
            # Construct sine signal
            sine = A * np.cos(2 * np.pi * t + phase * np.pi)
            sine -= sine.min()
            sine /= sine.max()
            self.sines.append(sine)

            # Construct label
            label = torch.zeros(n_samples)
            label[idx] = 1
            self.labels.append(label)
        
    def __len__(self):
        return len(self.sines)
    
    def __getitem__(self, idx):
        # Sine
        sine = self.sines[idx]
        if self.data_transform:
            sine = self.data_transform(sine)
        if self.encoder:
            sine = self.encoder(sine)
        
        # Label
        label = self.labels[idx]
        if self.label_transform:
            label = self.label_transform(label)
        if self.label_encoder:
            label = self.label_encoder(label)
        return {"input": sine, "label": label}

    def plot_sines(self):
        for i in range(len(self.sines)):
            sample = self.__getitem__(i)
            sine = sample["input"].sum((0, 1, 2))
            plt.plot(sine.numpy(), label="{}".format(i))
        plt.legend()


############################
# N-Caltech 101
############################
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


############################
# N-MNIST
############################
class NMNISTDataset(Dataset):
    def __init__(self, dataset_path, sample_file, sampling_time, sample_length):
        self.path = dataset_path 
        self.samples = np.loadtxt(sample_file).astype('int')
        self.sampling_time = sampling_time
        self.n_time_bins = int(sample_length / sampling_time)

    def __getitem__(self, index):
        input_index  = self.samples[index, 0]
        class_label  = self.samples[index, 1]
        
        input_spikes = read2Dspikes(self.path + str(input_index.item()) + '.bs2')
        input_spikes = input_spikes.toSpikeTensor(torch.zeros((2,34,34,self.n_time_bins)), 
                                                  samplingTime=self.sampling_time)
        desired_class = torch.zeros((10, 1, 1, 1))
        desired_class[class_label,...] = 1
        return input_spikes, desired_class, class_label
    
    def __len__(self):
        return self.samples.shape[0]


class NMNISTDatasetList(Dataset):
    def __init__(self, dataset_path, sample_file, sampling_time, sample_length, transform=None):
        self.path = dataset_path 
        self.samples = np.loadtxt(sample_file).astype('int')
        self.sampling_time = sampling_time
        self.n_time_bins = int(sample_length / sampling_time)
        self.transform = transform

    def __getitem__(self, index):
        input_index  = self.samples[index, 0]
        class_label  = self.samples[index, 1]
        
        input_spikes = read2Dspikes(self.path + str(input_index.item()) + '.bs2')
        input_spikes = torch.tensor([input_spikes.y, input_spikes.x, input_spikes.t, input_spikes.p]).t()
        input_spikes = input_spikes.to(torch.int32)
        input_spikes = input_spikes[input_spikes[:, 2] < 300]

        desired_class = torch.zeros((10, 1, 1, 1))
        desired_class[class_label,...] = 1

        if self.transform:
            input_spikes = self.transform(input_spikes)

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
        image = read2Dspikes(im_name)
        if len(image.x) > len(image.y):
            image.x = image.x[:len(image.y)]
        image = image.toSpikeTensor(torch.zeros(2,100,120,self.n_time_bins),
                                    samplingTime=self.sampling_time)

        label = self.im_info.iloc[idx, 1] 
            
        sample = {'image': image, 'label': label}
        return sample


class NCarsDatasetOld(Dataset):
    def __init__(self, root_dir, im_transform=None, lbl_transform=None):
        # Construct dataframe with image names and labels
        car_names = os.listdir(os.path.join(root_dir, 'cars_p'))
        car_names = [os.path.join('cars_p', car) for car in car_names]
        car_labels = np.ones(len(car_names), dtype=int)
        background_names = os.listdir(os.path.join(root_dir, 'background_p'))
        background_names = [os.path.join('background_p', backgr) for backgr in background_names]
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
        
    def __len__(self):
        return len(self.im_info)
    
    def __getitem__(self, idx):
        # Load image & label
        im_name = os.path.join(self.root_dir, self.im_info.iloc[idx, 0])
        with open(im_name, 'rb') as f:
            image = pickle.load(f)
        label = self.im_info.iloc[idx, 1] 
            
        # Apply transforms
        if self.im_transform:
            image = self.im_transform(image)
        if self.lbl_transform:
            label = self.lbl_transform(label)
            
        sample = {'image': image, 'label': label}
        return sample


############################
# Transforms
############################
class ToEventSpikeTensor():
    def __init__(self, resolution, t_bins):
        assert isinstance(resolution, tuple)
        self.t_bins = t_bins
        self.p_bins = 2
        self.h_bins = resolution[0]
        self.w_bins = resolution[1]
        
    def __call__(self, sample):
        assert sample.shape[1] == 4, 'Incorrect number of columns for event stream'
        im = np.zeros((self.p_bins, self.t_bins, self.h_bins, self.w_bins))
        sample[:, 0] = sample[:, 0] - sample[:, 0].min()
        sample[:, 0] = (sample[:, 0] / sample[:, 0].max()) * (self.t_bins - 1)
        for idx in range(sample.shape[0]):
            t = sample[idx, 0]
            w = int(sample[idx, 1])
            h = int(sample[idx, 2])
            p = int(sample[idx, 3])
            p = 0 if p == -1 else 1
            t_ind = np.floor(t).astype(int)
            im[p, t_ind, h, w] = t - t_ind
        im = im.reshape(-1, self.h_bins, self.w_bins)
        return im 

    
class ToTwoChannelImage():
    def __init__(self, im_size):
        assert isinstance(im_size, (int, tuple))
        self.p_bins = 2
        self.h_bins = im_size[0]
        self.w_bins = im_size[1]
        
    def __call__(self, sample):
        assert sample.shape[1] == 4, 'Incorrect number of columns for event stream'
        im = np.zeros((self.p_bins, self.h_bins, self.w_bins))
        for idx in range(sample.shape[0]):
            t = int(sample[idx, 0])
            w = int(sample[idx, 1])
            h = int(sample[idx, 2])
            p = int(sample[idx, 3])
            p = 0 if p == -1 else 1
            im[p, h, w] = t
        return im 


class ToTensor():
    def __call__(self, sample):
        if isinstance(sample, np.ndarray):
            return torch.from_numpy(sample).float()
        elif isinstance(sample, np.int64):
            return torch.Tensor([sample]).long()


class NumpyToTensor():
    def __call__(self, array):
        array = torch.from_numpy(array).float()
        array = array[(None,)*2]
        return array

class TensorToNumpy():
    def __call__(self, array):
        if isinstance(array, dict):
            for key, value in array.items():
                if isinstance(value, torch.Tensor):
                    array[key] = value.numpy()
        else:
            array = array.numpy()
        return array

class DiscretizeFloat():
    def __call__(self, tensor):
        tensor = tensor.int()
        return tensor.float()

class RepeatLabels():
    def __init__(self, n_repeat):
        self.n_repeat = n_repeat
        
    def __call__(self, sample):
        return sample.repeat(self.n_repeat)

class ToEmbeddingId():
    def __init__(self, h, v):
        self.h = h
        self.v = v
        
    def __call__(self, sample):
        sample -= 1  # Correction for matlab 1 indexing
        length = sample.shape[0]
        polarity = torch.Tensor([sample[:, 3]]).float()
        time = torch.Tensor([sample[:, 0]]).float()
        
        embedding = sample[:, 2] * self.h + sample[:, 1]
        embedding = torch.Tensor([embedding]).long()
        
        sample = {'time': time, 'embedding': embedding, 'polarity': polarity}
        return sample
    
class PadLength():
    def __init__(self, target_len):
        self.target_len = target_len
        
    def __call__(self, sample):
        length = sample['time'].shape[1]
        embedding = torch.zeros(self.target_len, 1, dtype=torch.int64)
        time = torch.zeros(self.target_len, 1)
        polarity = torch.zeros(self.target_len, 1)
        mask = torch.zeros(self.target_len, 1, dtype=torch.uint8)  # Elements that should be zero will not be masked!
        
        embedding[:length, 0] = sample['embedding']
        time[:length, 0] = sample['time']
        polarity[:length, 0] = sample['polarity']
        mask[length:, 0] = 1
        
        sample = {'time': time, 'embedding': embedding, 'polarity': polarity, 'mask': mask}
        return sample


class PadLengthNumpy():
    def __init__(self, target_len):
        self.target_len = target_len
        
    def __call__(self, sample):
        length = sample.shape[0]
        events = np.zeros((self.target_len, sample.shape[1]))
        events[:length, :] = sample
        return events


############################
# Transforms
############################
def plstm_collate(batch):
    embeddings, time, polarities, masks, labels = [], [], [], [], []
    for b in batch:
        sample, label = b
        embeddings.append(sample['embedding'])
        time.append(sample['time'])
        polarities.append(sample['polarity'])
        masks.append(sample['mask'])
        labels.append(label)
        
    embeddings = pad_sequence(embeddings).squeeze(2)
    time = pad_sequence(time)
    polarities = pad_sequence(polarities)
    masks = pad_sequence(masks).squeeze(2)
    labels = torch.tensor(labels, dtype=torch.int64)
    
#     embeddings = pack_padded_sequence(embeddings, lengths, enforce_sorted=False)
#     time = pack_padded_sequence(time, lengths, enforce_sorted=False)
#     polarities = pack_padded_sequence(polarities, lengths, enforce_sorted=False)
    
    samples = {'embedding': embeddings, 'time': time, 'polarity': polarities, 'mask': masks}
    return samples, labels
