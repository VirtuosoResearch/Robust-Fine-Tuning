import argparse
import collections
import math
import os
import random

import datasets
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, load_metric

import transformers
from data_loader.load_data_fns import label_noise, load_glue_tasks
from parse_config import ConfigParser
from trainer import *
from transformers import (AdamW, AutoModelForSequenceClassification,
                          BertConfig, BertForSequenceClassification,
                          SchedulerType, get_scheduler)
from utils import deep_copy, prepare_device
from utils.dual_t import (compose_T_matrices, est_t_matrix,
                          get_transition_matrices)


def init_weights(model):
    for name, module in model.named_modules():
        """Initialize the weights: Keep the pre-trained embeddings"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(config, args):
    logger = config.get_logger('train')
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    
    # Load dataset
    train_data_loader, valid_data_loader, test_data_loader, transformers_config = load_glue_tasks(
        args.task_name, logger=logger,
        model_name_or_path=args.model_name_or_path,
        pad_to_max_length=config["data_loader"]["args"]["pad_to_max_length"],
        max_length=config["data_loader"]["args"]["max_length"],
        train_batch_size=config["data_loader"]["args"]["train_batch_size"],
        eval_batch_size=config["data_loader"]["args"]["eval_batch_size"],
        noise_label=args.noise_rate, keep_index=args.keep_index
    )
    
    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)

    # Load pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=transformers_config,
    )
    if args.random_init:
        init_weights(model)
        logger.info("Initilize the weights of model")

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    source_state_dict = deep_copy(model.state_dict())

    runs = 1 if args.random_init else args.runs
    metrics = {}
    for i in range(runs):
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config["optimizer"]["args"]["weight_decay"],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config["optimizer"]["args"]["lr"])

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_data_loader) / config["trainer"]["gradient_accumulation_steps"])
        if config["trainer"]["max_train_steps"] == -1:
            config["trainer"]["max_train_steps"] = config["trainer"]["num_train_epochs"] * num_update_steps_per_epoch
        else:
            config["trainer"]["num_train_epochs"] = math.ceil(config["trainer"]["max_train_steps"] / num_update_steps_per_epoch)
        
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=config["trainer"]["num_warmup_steps"]*num_update_steps_per_epoch,
            num_training_steps=config["trainer"]["max_train_steps"],
        )
        
        if args.constraint_reweight:
            checkpoint_dir = os.path.join("saved", 
                "{}_{}_{}_{}_noise_rate_{}_constraint_reweight".format(
                    args.task_name, args.random_init, args.reg_method, config['optimizer']['args']['weight_decay'],
                    args.noise_rate
                ))
            trainer = ConstraintReweightTrainer(model, metric, optimizer, lr_scheduler,
                            config=config,
                            device=device,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            checkpoint_dir = checkpoint_dir,
                            noise_rate = args.reweight_noise_rate,
                            reweight_epoch = args.reweight_epoch,
                            num_classes=transformers_config.num_labels
                            )
            if args.reg_method == "constraint":
                trainer.add_constraint(
                    norm = args.reg_norm, 
                    lambda_attention = args.reg_attention, 
                    lambda_linear=args.reg_linear, 
                    lambda_pred_head=args.reg_predictor, 
                    state_dict=source_state_dict
                )
            if args.reg_method == "penalty":
                trainer.add_penalties(
                    norm = args.reg_norm, 
                    lambda_extractor = args.reg_penalty_encoder, 
                    lambda_pred_head=args.reg_penalty_predictor, 
                    state_dict = source_state_dict
                )
        else:
            checkpoint_dir = os.path.join("saved", 
                "{}_{}_{}_{}_noise_rate_{}".format(
                    args.task_name, args.random_init, args.reg_method, config['optimizer']['args']['weight_decay'],
                    args.noise_rate
                ))
            trainer = ConstraintGLUETrainer(model, metric, optimizer, lr_scheduler,
                            config=config,
                            device=device,
                            train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            test_data_loader=test_data_loader,
                            checkpoint_dir = checkpoint_dir
                            )
            if args.reg_method == "constraint":
                trainer.add_constraint(
                    norm = args.reg_norm, 
                    lambda_attention = args.reg_attention, 
                    lambda_linear=args.reg_linear, 
                    lambda_pred_head=args.reg_predictor, 
                    state_dict=source_state_dict
                )
            if args.reg_method == "penalty":
                trainer.add_penalties(
                    norm = args.reg_norm, 
                    lambda_extractor = args.reg_penalty_encoder, 
                    lambda_pred_head=args.reg_penalty_predictor, 
                    state_dict = source_state_dict
                )

        log = trainer.train()
        test_log = trainer.test()
        log.update(**{'test_'+k : v for k, v in test_log.items()})
        
        model.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "model_epoch_0.pth"))['state_dict']
        )
        for key, val in log.items():
            if key in metrics:
                metrics[key].append(val)
            else:
                metrics[key] = [val, ]
    for key, vals in metrics.items():
        logger.info("{}: {} +/- {}".format(key, np.mean(vals), np.std(vals)))

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    args.add_argument('--task_name', type=str, default="mrpc")
    args.add_argument('--model_name_or_path', type=str, default="bert-base-cased")
    args.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    args.add_argument("--random_init", action="store_true")
    args.add_argument("--runs", type=int, default=3)

    args.add_argument("--reg_method", type=str, default="none")
    args.add_argument("--reg_norm", type=str, default="frob")
    args.add_argument("--reg_attention", type=float, default=1.0 )
    args.add_argument("--reg_linear", type=float, default=1.0)
    args.add_argument("--reg_predictor", type=float, default=1.0)
    args.add_argument("--reg_penalty_encoder", type=float, default=1.0)
    args.add_argument("--reg_penalty_predictor", type=float, default=1.0)

    args.add_argument('--noise_rate', type=float, default=0.0)
    args.add_argument('--keep_index', action="store_true")

    args.add_argument('--constraint_reweight', action="store_true")
    args.add_argument('--reweight_epoch', type=int, default=1)
    args.add_argument('--reweight_noise_rate', type=float, default=0.8)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--epochs'], type=int, target='trainer;num_train_epochs'),
        CustomArgs(['--warm_up'], type=int, target='trainer;num_warmup_steps'),
        CustomArgs(['--early_stop'], type=int, target='trainer;early_stop'),
        CustomArgs(['--weight_decay'], type=float, target='optimizer;args;weight_decay')
    ]
    config, args = ConfigParser.from_args(args, options)
    main(config, args)
