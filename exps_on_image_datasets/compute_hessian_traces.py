import os
import torch
import collections
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.loss import nll_loss
from parse_config import ConfigParser
import data_loader.data_loaders as module_data
import model.model as module_arch
import argparse
from utils import prepare_device, deep_copy
import random
from model.modeling_vit import VisionTransformer, CONFIGS
from utils.hessian import set_seed, compute_hessians_trace

def main(config, args):
    set_seed(0)
    logger = config.get_logger('generate')
    logger = config.get_logger('train')

    # setup data_loader instances
    if config["data_loader"]["type"] == "CaltechDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, idx_start = 0, img_num = 30, phase = "train")
        valid_data_loader = config.init_obj('data_loader', module_data, idx_start = 30, img_num = 20, phase = "val")
        test_data_loader = config.init_obj('data_loader', module_data, idx_start = 50, img_num = 20, phase = "test")
    elif config["data_loader"]["type"] == "AircraftsDataLoader" or config["data_loader"]["type"] == "DomainNetDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, phase = "train")
        valid_data_loader = config.init_obj('data_loader', module_data, phase = "val")
        test_data_loader = config.init_obj('data_loader', module_data, phase = "test")
    elif config["data_loader"]["type"] == "BirdsDataLoader" or \
        config["data_loader"]["type"] == "CarsDataLoader" or \
        config["data_loader"]["type"] == "DogsDataLoader" or \
        config["data_loader"]["type"] == "IndoorDataLoader" or \
        config["data_loader"]["type"] == "Cifar10DataLoader":
        train_data_loader = config.init_obj('data_loader', module_data, valid_split = 0.1, phase = "train")
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = config.init_obj('data_loader', module_data, phase = "test")
    elif config["data_loader"]["type"] == "FlowerDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data)
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = train_data_loader.split_test()
    elif config["data_loader"]["type"] == "AnimalAttributesDataLoader":
        train_data_loader = config.init_obj('data_loader', module_data)
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = train_data_loader.split_test()

    logger.info("Train Size: {} Valid Size: {} Test Size: {}".format(
        len(train_data_loader.sampler), 
        len(valid_data_loader.sampler), 
        len(test_data_loader.sampler)))

    if args.is_vit:
        vit_config = CONFIGS[args.vit_type]
        model = config.init_obj('arch', module_arch, config = vit_config, img_size = args.img_size, zero_head=True)
    else:
        model = config.init_obj('arch', module_arch, pretrained=True)

    file = os.path.join("./saved_label_noise/", args.checkpoint_dir)
    device, device_ids = prepare_device(config['n_gpu'])
    model.load_state_dict(
            torch.load(os.path.join(file, args.checkpoint_name+".pth"))["state_dict"]
        )
    model.to(device)

    trace_dir = "./traces/{}_{}_{}_layer_traces.npy".format(
        config["data_loader"]["type"], config["arch"]["type"], args.save_name
    )

    if os.path.exists(trace_dir):
        max_layer_trace = np.load(trace_dir)
    else:
        max_layer_trace = []

    sample_size = 0
    not_improving = 0
    model.eval()
    for data, target, index in test_data_loader:
        num_samples = data.shape[0]
        layer_traces = compute_hessians_trace(model, nll_loss, data, target, device=device)

        if max_layer_trace == []:
            max_layer_trace = layer_traces
        else:
            max_layer_trace = np.maximum(max_layer_trace, layer_traces)
        
        logger.info(max_layer_trace)
        logger.info("========== Batch Complete ==========")

        np.save(trace_dir, max_layer_trace)

        sample_size += num_samples
        if sample_size > args.sample_size:
            break
        

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="cpu", type=str,
                      help='indices of GPUs to enable (default: all)')
    
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
    args.add_argument("--save_name", type=str, default="finetuned_train")
    args.add_argument("--sample_size", type=int, default=100)
    args.add_argument("--num_layers", type=int, default=18)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--model'], type=str, target="arch;type"),
        CustomArgs(['--domain'], type=str, target="data_loader;args;domain"),
        CustomArgs(['--sample'], type=int, target="data_loader;args;sample"),
    ]
    config, args = ConfigParser.from_args(args, options)
    print(config)
    main(config, args)