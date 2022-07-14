''' This file is used for calculating the sensitivity values of a network '''
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from model.model import ResNet101, ResNet18
import data_loader.data_loaders as data_loaders
import data_loader.data_loaders as module_data
import model.model as module_arch
from utils.util import deep_copy
from model.loss import nll_loss
import collections
import argparse
from parse_config import ConfigParser
import os
from utils import prepare_device
from collections import OrderedDict

from utils.hessian import compute_hessians_trace

''' Define a function to calculate stability '''
def perturbe_model_weights(state_dict, eps=0.001):
    for key, value in state_dict.items():
        if ("conv" in key or "pred_head" in key) and ("weight" in key) and (len(value.size())!=1):
            state_dict[key] += torch.randn_like(value)*eps
    return state_dict

def compute_loss(model, data_loader, device = "cpu"):
    loss = 0
    batch_count = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for idx, (data, labels, index) in enumerate(data_loader):
            data, labels = data.to(device), labels.to(device)

            loss += nll_loss(model(data), labels)
            batch_count += 1

    return loss/batch_count

def calculate_stability(model, data_loader, eps=1e-3, device = "cpu", runs = 20):
    ''' Calculate pred_vectors for model before perturbation '''
    loss_before = compute_loss(model, data_loader, device=device)
    state_dict_before = deep_copy(model.state_dict())
    print(f"Loss before: {loss_before}")

    '''
    Calculate the perturbed loss
    '''
    differences = []
    for i in range(runs):
        state_dict_after = deep_copy(state_dict_before)
        state_dict_after = perturbe_model_weights(state_dict_after, eps = eps)
        model.load_state_dict(state_dict_after)
        
        loss_after = compute_loss(model, data_loader, device=device)
        differece = loss_after - loss_before
        print(f"Loss after: {loss_after}")
        differences.append(differece.cpu().item())
    return differences

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(config, args):
    set_seed(0)
    logger = config.get_logger('generate')
    # setup data_loader instances
    if config["data_loader"]["type"] == "CaltechDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, idx_start = 0, img_num = 30, phase = "train")
        valid_data_loader = config.init_obj('data_loader', module_data, idx_start = 30, img_num = 20, phase = "val")
        test_data_loader = config.init_obj('data_loader', module_data, idx_start = 50, img_num = 20, phase = "test")
    elif config["data_loader"]["type"] == "AircraftsDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, phase = "train")
        valid_data_loader = config.init_obj('data_loader', module_data, phase = "val")
        test_data_loader = config.init_obj('data_loader', module_data, phase = "test")
    elif config["data_loader"]["type"] == "BirdsDataLoader" or \
        config["data_loader"]["type"] == "CarsDataLoader" or \
        config["data_loader"]["type"] == "DogsDataLoader" or \
        config["data_loader"]["type"] == "IndoorDataLoader" or \
        config["data_loader"]["type"] == "Cifar10DataLoader" or \
        config["data_loader"]["type"] == "Cifar100DataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, valid_split = 0.1, phase = "train")
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = config.init_obj('data_loader', module_data, phase = "test")
    elif config["data_loader"]["type"] == "FlowerDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data)
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = train_data_loader.split_test()
    logger.info("Train Size: {} Valid Size: {} Test Size: {}".format(
        len(train_data_loader.sampler), 
        len(valid_data_loader.sampler), 
        len(test_data_loader.sampler)))

    model = config.init_obj('arch', module_arch, pretrained=True)
    device, device_ids = prepare_device(config['n_gpu'])

    ''' Load model in difference phase of training and Plot their stability'''

    file = os.path.join("./saved_label_noise", args.checkpoint_dir)
    model.load_state_dict(
            torch.load(os.path.join(file, args.checkpoint_name+".pth"), map_location="cpu")["state_dict"]
        )

    data_loader = test_data_loader
    if args.sample_size < len(data_loader.sampler.indices):
        data_loader.sampler.indices = data_loader.sampler.indices[:args.sample_size]
    diff_losses = calculate_stability(model, data_loader, eps=args.eps, device = device)
    logger.info("Noise stability of {}: {:.4f} +/- {:.4f}".format(
        args.eps, np.mean(diff_losses), np.std(diff_losses)
    ))

    model.eval()
    if args.compute_hessian_trace:
        traces = []
        for data, target, index in data_loader:
            model.load_state_dict(
                torch.load(os.path.join(file, args.checkpoint_name+".pth"), map_location="cpu")["state_dict"]
            )
            
            layer_traces = compute_hessians_trace(model, nll_loss, data, target, device=device)
            
            traces.append(np.sum(layer_traces))
            
            logger.info(layer_traces)
            logger.info("sum of trace: {}".format(np.sum(layer_traces)))
            logger.info("sum of hessian traces: {}".format(np.mean(traces)))
            

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="cpu", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--data_frac', type=float, default=1.0)
    
    args.add_argument('--is_vit', action="store_true")
    args.add_argument('--img_size', type=int, default=224)
    args.add_argument("--vit_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    args.add_argument("--vit_pretrained_dir", type=str, default="checkpoints/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")

    args.add_argument("--checkpoint_dir", type=str, default="ResNet50_IndoorDataLoader_none_none_1.0000_1.0000_rand_init_True")
    args.add_argument("--checkpoint_name", type=str, default="model_epoch_30")

    args.add_argument("--sample_size", type=int, default=1000)
    args.add_argument("--eps", type=float, default=1e-3)
    args.add_argument("--compute_hessian_trace", action="store_true")

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--model'], type=str, target="arch;type"),
    ]
    config, args = ConfigParser.from_args(args, options)
    print(config)
    main(config, args)
