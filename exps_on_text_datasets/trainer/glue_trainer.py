import os
import numpy as np
import numpy
import torch
from numpy import inf
from utils import MetricTracker, prepare_inputs
from utils.constraint import FrobeniusConstraint, LInfLipschitzConstraint, add_penalty
from collections import OrderedDict
import torch.nn.functional as F
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class GLUETrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, metric, optimizer, lr_scheduler,
                 config, device,
                 train_data_loader, 
                 valid_data_loader=None,
                 test_data_loader=None, 
                 checkpoint_dir=None,
                 criterion=None):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.device = device

        self.model = model.to(device)
        self.criterion = criterion
        self.metric = metric
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.cfg_trainer = config['trainer']
        self.epochs = self.cfg_trainer['num_train_epochs']
        self.save_period = self.cfg_trainer['save_period']
        self.monitor = self.cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = self.cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1
        self.completed_steps = 0
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir is None:
            self.checkpoint_dir = config.save_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        else:
            for filename in os.listdir(self.checkpoint_dir):
                os.remove(os.path.join(self.checkpoint_dir, filename))

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.len_epoch = len(self.train_data_loader)

        self.train_metrics = MetricTracker('loss')
        self.valid_metrics = MetricTracker('loss')

        self._save_checkpoint(epoch=0, name="model_epoch_0")

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for step, batch in enumerate(self.train_data_loader):
            batch = prepare_inputs(batch, self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss = loss / self.cfg_trainer["gradient_accumulation_steps"]
            loss.backward()
            if step % self.cfg_trainer["gradient_accumulation_steps"] == 0 or step == len(self.train_data_loader) - 1:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.completed_steps += 1

            if self.completed_steps >= self.cfg_trainer["max_train_steps"]:
                break

            # update training metrics
            self.train_metrics.update('loss', loss.item())
            
            predictions = outputs.logits.argmax(dim=-1)
            self.metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )

        log = self.train_metrics.result()
        train_metrics = self.metric.compute()
        log.update(**train_metrics)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        for step, batch in enumerate(self.valid_data_loader):
            batch = prepare_inputs(batch, self.device)
            outputs = self.model(**batch)
            predictions = outputs.logits.argmax(dim=-1)

            self.valid_metrics.update('loss', outputs.loss.item())
            self.metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )
        
        log = self.valid_metrics.result()
        eval_metrics = self.metric.compute()
        log.update(**eval_metrics)
        return log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if improved:
                self._save_checkpoint(epoch)
        return log

    def test(self):
        if self.test_data_loader is None:
            self.logger.info("No test data set.")
            return

        # self.model = self.model.from_pretrained(self.checkpoint_dir).to(self.device)
        best_path = os.path.join(self.checkpoint_dir, f'model_best.pth')
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device)["state_dict"])

        self.model.eval()
        total_loss = 0.0
        for step, batch in enumerate(self.test_data_loader):
            batch = prepare_inputs(batch, self.device)
            outputs = self.model(**batch)
            predictions = outputs.logits.argmax(dim=-1)

            total_loss += outputs.loss.item() * self.config["data_loader"]["args"]["test_batch_size"]
            self.metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )

        n_samples = len(self.test_data_loader.dataset)
        log = {'loss': total_loss / n_samples}
        eval_metrics = self.metric.compute()
        log.update(**eval_metrics)
        self.logger.info(log)
        return log

    # old version
    def _save_checkpoint(self, epoch, name = "model_best"):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
        }
        # self.logger.info("Best checkpoint in epoch {}".format(epoch))
        
        best_path = os.path.join(self.checkpoint_dir, f'{name}.pth')
        torch.save(state, best_path)
        self.logger.info(f"Saving current model: {name}.pth ...")

    # old version
    def load_best_model(self):
        # Load the best model then test
        arch = type(self.model).__name__
        best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
        state_dict  = torch.load(best_path, map_location=self.device)["state_dict"]
        self.model.load_state_dict(state_dict)
        return state_dict

class ConstraintGLUETrainer(GLUETrainer):

    def __init__(self, model, metric, optimizer, lr_scheduler, config, device, 
        train_data_loader, valid_data_loader=None, test_data_loader=None, checkpoint_dir=None, criterion=None):
        super().__init__(model, metric, optimizer, lr_scheduler, config, device, 
        train_data_loader, valid_data_loader, test_data_loader, checkpoint_dir, criterion)
        self.constraints = []
        self.penalty = []

    def add_constraint(self, norm, 
        lambda_attention, lambda_linear, lambda_pred_head, state_dict = None):
        type_model = type(self.model)
        if norm == "inf-op":
            print("Do not support MARS norm now.")
        elif norm == "frob":
            self.constraints.append(
                FrobeniusConstraint(type_model, lambda_attention, 
                state_dict = state_dict, excluding_key = "LayerNorm", including_key="attention")
            )
            self.constraints.append(
                FrobeniusConstraint(type_model, lambda_linear, 
                state_dict = state_dict, excluding_key = "attention", including_key="encoder")
            )
            self.constraints.append(
                FrobeniusConstraint(type_model, lambda_pred_head, including_key = "classifier")
            )

    def add_penalties(self, norm, lambda_extractor, lambda_pred_head, state_dict=None):
        self.penalty.append(
            {"norm": norm, 
            "_lambda": lambda_extractor,
            "excluding_key": "LayerNorm",
            "including_key": "encoder",
            "state_dict": state_dict}
        )
        self.penalty.append(
            {"norm": norm, 
            "_lambda": lambda_pred_head,
            "excluding_key": None,
            "including_key": "classifier",
            "state_dict": None}
        )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for step, batch in enumerate(self.train_data_loader):
            batch = prepare_inputs(batch, self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss = loss / self.cfg_trainer["gradient_accumulation_steps"]

            """Apply Penalties"""
            for penal in self.penalty:
                loss += add_penalty(
                    self.model, 
                    penal["norm"], 
                    penal["_lambda"], 
                    excluding_key = penal["excluding_key"],
                    including_key = penal["including_key"],
                    state_dict=penal["state_dict"]
                )
            """Apply Penalties"""

            loss.backward()
            if step % self.cfg_trainer["gradient_accumulation_steps"] == 0 or step == len(self.train_data_loader) - 1:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.completed_steps += 1

                """Apply Constraints"""
                for constraint in self.constraints:
                    self.model.apply(constraint)
                """Apply Constraints"""

            if self.completed_steps >= self.cfg_trainer["max_train_steps"]:
                break

            # update training metrics
            self.train_metrics.update('loss', loss.item())
            
            predictions = outputs.logits.argmax(dim=-1)
            self.metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )

        log = self.train_metrics.result()
        train_metrics = self.metric.compute()
        log.update(**train_metrics)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        return log

class ConstraintReweightTrainer(ConstraintGLUETrainer):

    def __init__(self, model, metric, optimizer, lr_scheduler, config, device, 
        train_data_loader, valid_data_loader=None, test_data_loader=None, checkpoint_dir=None, criterion=None,
        noise_rate = 0.0, reweight_epoch = 0, num_classes = 2):
        super().__init__(model, metric, optimizer, lr_scheduler, config, device, 
        train_data_loader, valid_data_loader, test_data_loader, checkpoint_dir, criterion)

        self.reweight_epoch = reweight_epoch
        self.noise_rate = noise_rate
        self.num_classes = num_classes
        self.genereate_confusion_matrix()

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
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for step, batch in enumerate(self.train_data_loader):
            batch = prepare_inputs(batch, self.device)
            outputs = self.model(**batch)
            if epoch < self.reweight_epoch:
                loss = outputs.loss
            else:
                log_probs = F.log_softmax(outputs.logits, dim=1)
                labels = batch['labels']
                loss = self.reweighted_loss(log_probs, labels)
            loss = loss / self.cfg_trainer["gradient_accumulation_steps"]

            """Apply Penalties"""
            for penal in self.penalty:
                loss += add_penalty(
                    self.model, 
                    penal["norm"], 
                    penal["_lambda"], 
                    excluding_key = penal["excluding_key"],
                    including_key = penal["including_key"],
                    state_dict=penal["state_dict"]
                )
            """Apply Penalties"""

            loss.backward()
            if step % self.cfg_trainer["gradient_accumulation_steps"] == 0 or step == len(self.train_data_loader) - 1:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.completed_steps += 1

                """Apply Constraints"""
                for constraint in self.constraints:
                    self.model.apply(constraint)
                """Apply Constraints"""

            if self.completed_steps >= self.cfg_trainer["max_train_steps"]:
                break

            # update training metrics
            self.train_metrics.update('loss', loss.item())
            
            predictions = outputs.logits.argmax(dim=-1)
            self.metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )

        log = self.train_metrics.result()
        train_metrics = self.metric.compute()
        log.update(**train_metrics)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        return log
