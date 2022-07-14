import torch
import numpy as np
import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)

def custom_kl_div(prediction, target):
    output_pos = target * (target.clamp(min=1e-7).log() - prediction)
    zeros = torch.zeros_like(output_pos)
    output = torch.where(target > 0, output_pos, zeros)
    output = torch.sum(output, axis=1)
    return output.mean()

class JensenShannonDivergenceWeightedScaled(torch.nn.Module):
    def __init__(self, num_classes, weights):
        super(JensenShannonDivergenceWeightedScaled, self).__init__()
        self.num_classes = num_classes
        self.weights = weights
        
        self.scale = -1.0 / ((1.0-self.weights[0]) * np.log((1.0-self.weights[0])))
        assert abs(1.0 - sum(self.weights)) < 0.001
    
    def forward(self, pred, labels):
        preds = list()
        
        preds.append(torch.exp(pred))

        labels = F.one_hot(labels, self.num_classes).float() 
        distribs = [labels] + preds
        assert len(self.weights) == len(distribs)

        mean_distrib = sum([w*d for w,d in zip(self.weights, distribs)])
        mean_distrib_log = mean_distrib.clamp(1e-7, 1.0).log()
        
        jsw = sum([w*custom_kl_div(mean_distrib_log, d) for w,d in zip(self.weights, distribs)])
        return self.scale * jsw