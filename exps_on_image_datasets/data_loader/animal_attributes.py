from torch.utils.data import Dataset
from .base_data_loader import BaseDataLoader
import torchvision.transforms as transforms
from copy import deepcopy
from torch.utils.data import DataLoader
import torch
import numpy as np
import os

DOMAIN_NET_DIR = './data/animal_attributes/'

DATA_DIR = os.path.join(DOMAIN_NET_DIR, "unseen_data.npy")
TRUE_LABEL_DIR = os.path.join(DOMAIN_NET_DIR, "unseen_labels.npy")
NOISY_LABEL_DIR = os.path.join(DOMAIN_NET_DIR, "predicted_labels.npy")

class AnimalAttributesDataset(Dataset):

    def __init__(self, transform):
        self.imgs = np.load(DATA_DIR) # unomarlized images resized to 224 and saved in numpy 
        self.true_labels = np.load(TRUE_LABEL_DIR)
        self.labels = np.load(NOISY_LABEL_DIR)
        self.transform = transform

    def __getitem__(self, index):
        img = self.transform(self.imgs[index])
        label = self.labels[index]
        return img, label, index
    
    def __len__(self):
        return len(self.imgs)

class AnimalAttributesDataLoader(BaseDataLoader):

    train_transform = transforms.Compose([
        lambda x: torch.Tensor(x),
        transforms.ToPILImage(),
		transforms.Resize((224, 224)),
        transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
	    ])

    test_transform = transforms.Compose([
        lambda x: torch.Tensor(x),
        transforms.ToPILImage(),
		transforms.Resize((224, 224)),
        transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
	    ])

    def __init__(self, batch_size, shuffle, valid_split, test_split, num_workers):

        self.dataset = AnimalAttributesDataset(transform=self.train_transform)
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, valid_split=valid_split, test_split=test_split, num_workers=num_workers)

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            kwargs = deepcopy(self.init_kwargs)
            tmp_dataset = kwargs["dataset"]
            tmp_dataset.transform = self.test_transform
            tmp_dataset.labels = tmp_dataset.true_labels
            return DataLoader(sampler=self.valid_sampler, **kwargs)

    def split_test(self):
        if self.test_sampler is None:
            return None
        else:
            kwargs = deepcopy(self.init_kwargs)
            tmp_dataset = kwargs["dataset"]
            tmp_dataset.transform = self.test_transform
            tmp_dataset.labels = tmp_dataset.true_labels
            return DataLoader(sampler=self.test_sampler, **kwargs)
