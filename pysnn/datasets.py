import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
import torchvision

from pysnn.file_io import Events, read_2d_spikes


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
            x_trn, x_tst, y_trn, y_tst = train_test_split(
                dset, labels[idx], train_size=train_size, random_state=seed
            )
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
    df = df[["sample", "label"]]
    return df, names


def train_test(root_dir, train_size=0.8, seed=42):
    r"""Split dataset into train and test sets.
    
    Takes in a directory where it looks for sub-directories. Content of each directory is split into train and test subsets.

    :param root_dir: Directory containing data.
    :param train_size: Percentage of the data to be assigned to training set, 1 - train_size is assigned as test set.
    :param seed: Seed for random number generator.
    """
    content = _list_dir_content(root_dir)
    data, _ = _concat_dir_content(content)
    train, test = _train_test_split_classification(
        data, train_size=train_size, seed=seed
    )
    return train, test


class NeuromorphicDataset(Dataset):
    r"""Class that wraps around several neuromorphic datasets.

    The class adheres to regular PyTorch dataset conventions.

    :param data: Data for dataset formatted in a pd.DataFrame.
    :param sampling_time: Duration of interval between samples.
    :param sample_length: Total duration of a single sample.
    :param height: Number of pixels in height direction.
    :param width: Number of pixels in width direction.
    :param im_transform: Image transforms, same convention as for PyTorch datasets.
    :param lbl_transform: Lable transforms, same convention as for PyTorch datasets.
    """

    def __init__(
        self,
        data,
        sampling_time,
        sample_length,
        height,
        width,
        im_transform=None,
        lbl_transform=None,
        lbl_encoder=None,
    ):
        self.data = data
        self.im_transform = im_transform
        self.lbl_transform = lbl_transform
        self.lbl_encoder = lbl_encoder

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
        sample = input_spikes.to_spike_tensor(
            self.im_template, sampling_time=self.sampling_time
        ).bool()

        # Label
        label = torch.tensor(self.data.iloc[idx, 1])
        class_idx = torch.tensor(self.data.iloc[idx, 1])

        # Apply transforms
        if self.im_transform:
            sample = self.im_transform(sample)
        if self.lbl_transform:
            label = self.lbl_transform(label)
        if self.lbl_encoder:
            label = self.lbl_encoder(label)

        return sample, label, class_idx, class_idx


#########################################################
# Neurmorphic Caltech 101
#########################################################
def ncaltech_train_test(
    root,
    sampling_time=1,
    sample_length=300,
    height=200,
    width=300,
    im_transform=None,
    lbl_transform=None,
):
    r"""Neurmorphic version of the Caltech-101 dataset, obtained from:

        https://www.garrickorchard.com/datasets/n-caltech101
    
    'Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades' by G. Orchard et al.

    :return: :class:`NeuromorphicDataset` for both and training and test data.
    """
    train, test = train_test(root)
    train_dataset = NeuromorphicDataset(
        train,
        sampling_time,
        sample_length,
        height,
        width,
        im_transform=im_transform,
        lbl_transform=lbl_transform,
    )
    test_dataset = NeuromorphicDataset(
        test,
        sampling_time,
        sample_length,
        height,
        width,
        im_transform=im_transform,
        lbl_transform=lbl_transform,
    )
    return train_dataset, test_dataset


#########################################################
# Regular MNIST in spiking format
#########################################################
def create_torchvision_dataset_wrapper(ds_type):
    # language=rst
    """
    Adapted from BindsNET https://github.com/BindsNET/bindsnet

    Creates wrapper classes for datasets that output ``(image, label)`` from
    ``__getitem__``. This applies to all of the datasets inside of ``torchvision``.
    """
    if type(ds_type) == str:
        ds_type = getattr(torchvision.datasets, ds_type)

    class TorchvisionDatasetWrapper(ds_type):
        __doc__ = (
            """BindsNET torchvision dataset wrapper for:
        The core difference is the output of __getitem__ is no longer
        (image, label) rather a dictionary containing the image, label,
        and their encoded versions if encoders were provided.
            \n\n"""
            + str(ds_type)
            if ds_type.__doc__ is None
            else ds_type.__doc__
        )

        def __init__(
            self,
            image_encoder=None,
            label_encoder=None,
            n_ims=None,
            *args,
            **kwargs
        ):
            # language=rst
            """
            Constructor for the BindsNET torchvision dataset wrapper.
            For details on the dataset you're interested in visit
            https://pytorch.org/docs/stable/torchvision/datasets.html
            :param image_encoder: Spike encoder for use on the image
            :param label_encoder: Spike encoder for use on the label
            :param *args: Arguments for the original dataset
            :param **kwargs: Keyword arguments for the original dataset
            """
            super().__init__(*args, **kwargs)

            if n_ims:
                idxs = torch.randperm(self.data.shape[0])[:n_ims]
                self.data = self.data[idxs]
                self.targets = self.targets[idxs]

            self.args = args
            self.kwargs = kwargs

            self.image_encoder = image_encoder
            self.label_encoder = label_encoder

        def __getitem__(self, idx):
            # language=rst
            """
            Utilizes the ``torchvision.dataset`` parent class to grab the data, then
            encodes using the supplied encoders.
            :param int idx: Index to grab data at.
            :return: The relevant data and encoded data from the requested index.
            """
            image, label = super().__getitem__(idx)
            label = torch.tensor(label, dtype=torch.int64)
            label_int = label.clone()

            if self.image_encoder:
                image = self.image_encoder(image)
            if self.label_encoder:
                label = self.label_encoder(label)

            return image, label, label_int, label_int

        def __len__(self):
            return super().__len__()

    return TorchvisionDatasetWrapper

mnist_wrapper = create_torchvision_dataset_wrapper("MNIST")


#########################################################
# Neuromorphic MNIST
#########################################################
def nmnist_train_test(
    root,
    train_samples=None,
    test_samples=None,
    sampling_time=1,
    sample_length=300,
    height=34,
    width=34,
    im_transform=None,
    lbl_transform=None,
    lbl_encoder=None,
):
    r"""Neurmorphic version of the MNIST dataset, obtained from:

        https://www.garrickorchard.com/datasets/n-mnist
    
    'Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades' by G. Orchard et al.

    :return: :class:`NeuromorphicDataset` for both and training and test data.
    """
    if not train_samples:
        train_content = _list_dir_content(os.path.join(root, "Train"))
        train, _ = _concat_dir_content(train_content)
    else:
        train, _ = train_test(
            os.path.join(root, "Train"), train_size=(train_samples / 60000)
        )
    train_dataset = NeuromorphicDataset(
        train,
        sampling_time,
        sample_length,
        height,
        width,
        im_transform=im_transform,
        lbl_transform=lbl_transform,
        lbl_encoder=lbl_encoder,
    )

    if not test_samples:
        test_content = _list_dir_content(os.path.join(root, "Test"))
        test, _ = _concat_dir_content(test_content)
    else:
        _, test = train_test(os.path.join(root, "Test"), train_size=((10000 - test_samples) / 10000))
    test_dataset = NeuromorphicDataset(
        test,
        sampling_time,
        sample_length,
        height,
        width,
        im_transform=im_transform,
        lbl_transform=lbl_transform,
        lbl_encoder=lbl_encoder,
    )

    return train_dataset, test_dataset


#########################################################
# Neuromorphic MNIST
#########################################################
def ncars_train_test(
    root,
    sampling_time=1,
    sample_length=100,
    height=100,
    width=120,
    im_transform=None,
    lbl_transform=None,
):
    r"""Neurmorphic dataset containing images of cars or background, obtained from:

        https://www.prophesee.ai/dataset-n-cars/
    
    This is a two class problem.
    
    :return: :class:`NeuromorphicDataset` for both and training and test data.
    """
    train_content = _list_dir_content(os.path.join(root, "train"))
    train, _ = _concat_dir_content(train_content)
    train_dataset = NeuromorphicDataset(
        train,
        sampling_time,
        sample_length,
        height,
        width,
        im_transform=im_transform,
        lbl_transform=lbl_transform,
    )

    test_content = _list_dir_content(os.path.join(root, "test"))
    test, _ = _concat_dir_content(test_content)
    test_dataset = NeuromorphicDataset(
        test,
        sampling_time,
        sample_length,
        height,
        width,
        im_transform=im_transform,
        lbl_transform=lbl_transform,
    )

    return train_dataset, test_dataset


############################
# Boolean
############################
class Boolean(Dataset):
    r"""Dataset for generating event-based, boolean data samples. Can be used to construct AND, OR, XOR datasets.
    
    :param data_encoder: ``Encoder`` to convert scalar data values to spiketrains.
    :param data_transform: Image transforms, same convention as for PyTorch datasets.
    :param lbl_transform: Label transforms, same convention as for PyTorch datasets.
    :repeats: Number of times to repeat the 4 samples within a single iteration of the dataset, total is 4 * repeats.
    """

    def __init__(
        self,
        labels,
        data_encoder=None,
        lbl_encoder=None,
        data_transform=None,
        lbl_transform=None,
        repeats=1,
        sample_repeats=None,
    ):
        self.data_encoder = data_encoder
        self.lbl_encoder = lbl_encoder
        self.data_transform = data_transform
        self.lbl_transform = lbl_transform
        self.n_samples = 4

        # Generate data and labels
        self.data = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float)
        self.data = torch.repeat_interleave(self.data, int(repeats), dim=1)
        self.labels = labels
        self.classes = torch.arange(4).unsqueeze(1)

        if sample_repeats:
            self.data = self.data.repeat(sample_repeats, 1)
            self.labels = self.labels.repeat(sample_repeats, 1)
            self.classes = self.classes.repeat(sample_repeats, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].unsqueeze(0)
        label = self.labels[idx]
        sample_class = self.classes[idx]
        transformed_sample = None

        # Sample transforms
        if self.data_transform:
            sample = self.data_transform(sample)
            transformed_sample = sample.clone()
        if self.data_encoder:
            sample = self.data_encoder(sample)

        # Label transforms
        if self.lbl_transform:
            label = self.lbl_transform(label)
        if self.lbl_encoder:
            label = self.lbl_encoder(label)

        return sample, label, sample_class, transformed_sample


class XOR(Boolean):
    r"""XOR dataset, inherits directly from :class:`Boolean`."""

    def __init__(
        self,
        data_encoder=None,
        lbl_encoder=None,
        data_transform=None,
        lbl_transform=None,
        repeats=1,
        sample_repeats=None,
    ):
        classes = torch.tensor([[0], [1], [1], [0]])
        super(XOR, self).__init__(
            classes,
            data_encoder,
            lbl_encoder,
            data_transform,
            lbl_transform,
            repeats,
            sample_repeats,
        )


class AND(Boolean):
    r"""AND dataset, inherits directly from :class:`Boolean`."""

    def __init__(
        self,
        data_encoder=None,
        lbl_encoder=None,
        data_transform=None,
        lbl_transform=None,
        repeats=1,
        sample_repeats=None,
    ):
        classes = torch.tensor([[0], [0], [0], [1]])
        super(AND, self).__init__(
            classes,
            data_encoder,
            lbl_encoder,
            data_transform,
            lbl_transform,
            repeats,
            sample_repeats,
        )


class OR(Boolean):
    r"""OR dataset, inherits directly from :class:`Boolean`."""

    def __init__(
        self,
        data_encoder=None,
        lbl_encoder=None,
        data_transform=None,
        lbl_transform=None,
        repeats=1,
        sample_repeats=None,
    ):
        classes = torch.tensor([[0], [1], [1], [1]])
        super(OR, self).__init__(
            classes,
            data_encoder,
            lbl_encoder,
            data_transform,
            lbl_transform,
            repeats,
            sample_repeats,
        )


############################
# transforms
############################
class DiscretizeFloat:
    r"""Discretize float to nearest integer values, returns a torch.Float."""

    def __call__(self, tensor):
        tensor = tensor.int()
        return tensor.float()


class BooleanNoise:
    r"""Add noise to the integer boolean values, e.g. 1 -> 0.8 and 0 -> 0.2
    
    :param low_thresh: Upper boundary for the negative values.
    :param high_thresh: Lower boundary for the positive values.
    """

    def __init__(self, low_thresh, high_thresh):
        self.low_distr = torch.distributions.Uniform(0.0, low_thresh)
        self.high_distr = torch.distributions.Uniform(high_thresh, 1.0)

    def __call__(self, x):
        zero_idx = x == 0
        one_idx = x == 1
        if zero_idx.any():
            x[zero_idx] = self.low_distr.sample(zero_idx.shape)[zero_idx]
        if one_idx.any():
            x[one_idx] = self.high_distr.sample(one_idx.shape)[one_idx]
        return x


class Intensity:
    r"""Multiplication with a fixed scalar value."""

    def __init__(self, intensity):
        self.intensity = intensity

    def __call__(self, x):
        return x * self.intensity


if __name__ == "__main__":
    from torchvision.transforms import ToTensor

    # train, test = nmnist_train_test(os.path.expanduser("~/Thesis_final/data/nmnist/"))
    data = mnist_wrapper(n_ims=10, root="/home/bas/stack/ai_projects/graph_nns/data/", transform=ToTensor(), train=False)
    