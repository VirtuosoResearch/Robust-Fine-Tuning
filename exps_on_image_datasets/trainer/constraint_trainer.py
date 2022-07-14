from random import betavariate
from model.modeling_vit import VisionTransformer
import torch
import os

from .base_trainer import Trainer
import numpy as np
from utils.constraint import LInfLipschitzConstraint, FrobeniusConstraint, add_penalty, \
    LInfLipschitzConstraintRatio, FrobeniusConstraintRatio

import torch.nn.functional as F
from collections import OrderedDict

class ConstraintTrainer(Trainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, config, 
        device, train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir):
        super(ConstraintTrainer, self).__init__(model, criterion, metric_ftns, optimizer, config, 
        device, train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir)
        self.penalty = []
        self.constraints = []

        self._save_checkpoint(0, name="model_epoch_0.pth")

        self.num_classes = config["arch"]["args"]["n_classes"]
        # labels = train_data_loader.dataset.labels
        # self.predictions = torch.zeros(len(labels), self.num_classes, dtype=torch.float).to(device)

    def add_penalty(self, norm, lambda_extractor, lambda_pred_head, state_dict=None, scale_factor=1.0):
        self.penalty.append(
            {"norm": norm, 
             "_lambda": lambda_extractor,
             "excluding_key": "pred_head",
             "including_key": "layer1",
             "state_dict": state_dict}
        )
        self.penalty.append(
            {"norm": norm, 
             "_lambda": lambda_extractor,
             "excluding_key": "pred_head",
             "including_key": "layer2",
             "state_dict": state_dict}
        )
        self.penalty.append(
            {"norm": norm, 
             "_lambda": lambda_extractor*scale_factor,
             "excluding_key": "pred_head",
             "including_key": "layer3",
             "state_dict": state_dict}
        )
        self.penalty.append(
            {"norm": norm, 
             "_lambda": lambda_extractor*pow(scale_factor, 2),
             "excluding_key": "pred_head",
             "including_key": "layer4",
             "state_dict": state_dict}
        )
        self.penalty.append(
            {"norm": norm, 
             "_lambda": lambda_pred_head,
             "excluding_key": None,
             "including_key": "pred_head",
             "state_dict": None}
        )

    def add_constraint(self, norm, lambda_extractor, lambda_pred_head, state_dict = None, scale_factor = 1.0, use_ratio = False):
        '''
        Add hard constraint for model weights
            for feature_extractor, it will contraint the weight to pretrain weight
            for pred_head, it will contraint the weight to zero
        '''
        if use_ratio:
            # if use_ratio, lambda_extractor is a ratio between, lambda_pred_head is absolute distance
            if norm == "inf-op":
                self.constraints.append(
                    LInfLipschitzConstraintRatio(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_pred_head, 
                    including_key = "pred_head")
                )
            elif norm == "frob":
                self.constraints.append(
                    FrobeniusConstraintRatio(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_pred_head, including_key = "pred_head")
                )
        elif type(self.model) == VisionTransformer:
            if norm == "inf-op":
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="encoder")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_pred_head, 
                    including_key = "pred_head")
                )
            elif norm == "frob":
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="encoder")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_pred_head, including_key = "pred_head")
                )
        else:
            # is not use_ratio, then both the lambda_extractor & lambda_pred_head is absolute value; 
            # here we could use layer-wise distance
            if norm == "inf-op":
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer1")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer2")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_extractor*scale_factor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer3")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_extractor*pow(scale_factor, 2), 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer4")
                )
                self.constraints.append(
                    LInfLipschitzConstraint(type(self.model), lambda_pred_head, 
                    including_key = "pred_head")
                )
            elif norm == "frob":
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer1")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_extractor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer2")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_extractor*scale_factor, 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer3")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_extractor*pow(scale_factor, 2), 
                    state_dict = state_dict, excluding_key = "pred_head", including_key="layer4")
                )
                self.constraints.append(
                    FrobeniusConstraint(type(self.model), lambda_pred_head, including_key = "pred_head")
                )

    def save_predictions(self, output, target, index):
        probs = torch.exp(output.detach())
        self.predictions[index] = probs

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target, index) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            # self.save_predictions(output, target, index)
            loss = self.criterion(output, target)

            """Apply Penalties"""
            for penalty in self.penalty:
                loss += add_penalty(
                    self.model, 
                    penalty["norm"], 
                    penalty["_lambda"], 
                    excluding_key = penalty["excluding_key"],
                    including_key = penalty["including_key"],
                    state_dict=penalty["state_dict"]
                )
            """Apply Penalties"""

            loss.backward()
            self.optimizer.step()

            """Apply Constraints"""
            for constraint in self.constraints:
                self.model.apply(constraint)
            """Apply Constraints"""

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx == self.len_epoch:
                break
        # torch.save(self.predictions, os.path.join(self.checkpoint_dir, f'model_predictions_{epoch}.pth'))
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

class ConstraintReweightTrainer(ConstraintTrainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, 
    train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir,
    noise_rate = 0.0, reweight_epoch = 0, confusion_matrix = None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, device, 
        train_data_loader, valid_data_loader, test_data_loader, lr_scheduler, checkpoint_dir)

        self.reweight_epoch = reweight_epoch
        self.noise_rate = noise_rate
        self.num_classes = config["arch"]["args"]["n_classes"]
        if confusion_matrix is None:
            self.genereate_confusion_matrix()
        else:
            self.confusion_matrix = torch.tensor(confusion_matrix, device=self.device)
            self.reweight_matrix = torch.linalg.pinv(self.confusion_matrix)
            self.logger.info("Reweight matrix:")
            self.logger.info(self.reweight_matrix)
        assert hasattr(self, "reweight_matrix")

    def genereate_confusion_matrix(self):
        count_matrix = torch.zeros(self.num_classes, self.num_classes, device=self.device)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                count_matrix[i][j] = (1-self.noise_rate) if i == j else self.noise_rate/(self.num_classes-1)
        
        self.confusion_matrix = count_matrix / torch.sum(count_matrix, dim=1, keepdim=True)
        self.logger.info("Confusion matrix:")
        self.logger.info(self.confusion_matrix)
        self.reweight_matrix = torch.linalg.pinv(self.confusion_matrix)
        # self.reweight_matrix[self.reweight_matrix<0] = 0.0
        self.logger.info("Reweight matrix:")
        self.logger.info(self.reweight_matrix)

    def reweighted_loss(self, output, target):
        reweighted_loss = -output * self.reweight_matrix[target]
        return torch.sum(reweighted_loss, dim=-1).mean()# , self.criterion(output, target)

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target, index) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            if epoch < self.reweight_epoch:
                loss = self.criterion(output, target)                
            else:
                loss = self.reweighted_loss(output, target)

            """Apply Penalties"""
            for penalty in self.penalty:
                loss += add_penalty(
                    self.model, 
                    penalty["norm"], 
                    penalty["_lambda"], 
                    excluding_key = penalty["excluding_key"],
                    including_key = penalty["including_key"],
                    state_dict=penalty["state_dict"]
                )
            """Apply Penalties"""

            loss.backward()
            self.optimizer.step()

            """Apply Constraints"""
            for constraint in self.constraints:
                self.model.apply(constraint)
            """Apply Constraints"""

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx == self.len_epoch:
                break
        # torch.save(self.predictions, os.path.join(self.checkpoint_dir, f'model_predictions_{epoch}.pth'))
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log
