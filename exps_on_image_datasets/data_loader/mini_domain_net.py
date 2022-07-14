import pickle
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.utils.data as data
from .base_data_loader import BaseDataLoader
from .rand_augment import TransformFixMatch

DOMAIN_NET_DIR = './data/domain_net/'

def get_classes(sample=1):
    '''
    Function to read the list of sampled classes
    '''

    with open(DOMAIN_NET_DIR + "sample_classes_" + str(sample) + ".txt", "r") as f:
        l = f.readlines()
    
    return [x.strip() for x in l]


class MiniNoisyDomainNet(Dataset):
    def __init__(self, partition, domain, classes, sample, transform):
        super().__init__()
        self.partition = partition
        self.domain = domain
        self.transform = transform

        dir = DOMAIN_NET_DIR + "sample_" + str(sample)
        with open(os.path.join(dir, f'{partition}_{domain}.pkl'), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.imgs = data['data']
            
            labels = data['labels']

            # adjust sparse labels to labels from 0 to n.
            self.word2label = {x: i for i, x in enumerate(classes)}
            new_labels = np.array([self.word2label[name] for name in labels])

            self.true_labels = new_labels
            self.label2word = {v: k for k, v in self.word2label.items()}

        if partition == "train":
            self.labels = np.load(DOMAIN_NET_DIR + f"sample_{sample}/" + f'{partition}_{domain}_mv_labels.npy')
        else:
            self.labels = self.true_labels
    
    def get_label2word(self):
        return self.label2word
    
    def __getitem__(self, index):
        img = self.transform(self.imgs[index])
        label = self.labels[index]
        return img, label, index
    
    def __len__(self):
        return len(self.imgs)

class DomainNetDataLoader(BaseDataLoader):
    def __init__(self, data_dir, domain, sample, batch_size, shuffle=True, valid_split=0.0, num_workers=1, phase="train", strong_augment = False):
        classes = get_classes(sample)
        sample_dir = DOMAIN_NET_DIR + "sample_%d" % (sample)
        with open(os.path.join(sample_dir, f'train_{domain}.pkl'), 'rb') as f:
            # getting transform
            domain_dict = pickle.load(f, encoding='latin1')
            train_data = domain_dict["data"]
            
            # converting to 0-1 values
            train_data = train_data / 255.0
            mean = np.mean(train_data, axis=(0, 1, 2))
            std = np.std(train_data, axis=(0, 1, 2))
        if strong_augment:
            trsfm = transforms.Compose([
                lambda x: Image.fromarray(x),
                TransformFixMatch(size=224, mean=mean, std=std)
            ])
        else:
            trsfm = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        self.data_dir = data_dir
        self.dataset = MiniNoisyDomainNet(phase, domain, classes, sample, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, valid_split=valid_split, test_split=0, num_workers=num_workers)

def get_loaders(domain, sample, batch_size=50):
    '''
    Function to get train, val, and testloaders for given domain of DomainNet

    Args:
    domain - a string representing one of the domains
    '''

    img_size = (84, 84)
    classes = get_classes(sample)
    print(classes)

    sample_dir = DOMAIN_NET_DIR + "sample_%d" % (sample)
    with open(os.path.join(sample_dir, f'train_{domain}.pkl'), 'rb') as f:
        
        # getting transform
        domain_dict = pickle.load(f, encoding='latin1')
        train_data = domain_dict["data"]
        
        # converting to 0-1 values
        train_data = train_data / 255.0
        mean = np.mean(train_data, axis=(0, 1, 2))
        std = np.std(train_data, axis=(0, 1, 2))

    transform = transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transform = transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = MiniNoisyDomainNet("train", domain, classes, sample, transform=transform)
    trainloader = data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    val_dataset = MiniNoisyDomainNet("val", domain, classes, sample, transform=test_transform)
    valloader = data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    test_dataset = MiniNoisyDomainNet("test", domain, classes, sample, transform=test_transform)
    testloader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    
    return trainloader, valloader, testloader

    