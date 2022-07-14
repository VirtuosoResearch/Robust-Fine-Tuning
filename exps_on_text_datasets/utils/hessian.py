from utils.util import prepare_inputs
from collections import OrderedDict

import torch
import numpy as np
import random

def normalization(vs):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    norms = [torch.sum(v*v) for v in vs]
    norms = [(norm**0.5).cpu().item() for norm in norms]
    vs = [vi / (norms[i] + 1e-6) for (i, vi) in enumerate(vs)]
    return vs

def orthnormal(ws, vs_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for vs in vs_list:
        for w, v in zip(ws, vs):
            w.data.add_(-v*(torch.sum(w*v)))
    return normalization(ws)

""" Calculate Top Eigenvalue of Hessian """ 
def compute_eigenvalue(model, batch, device, maxIter=100, tol=1e-3, top_n=1):
    # Get parameters and gradients of corresponding layer
    batch = prepare_inputs(batch, device)
    outputs = model(**batch)
    loss = outputs.loss

    layers = get_layers(model)
    weights = [module.weight for name, module in layers.items()]
    model.zero_grad()
    """ use negative loss to get the minimum eigenvalue here """
    gradients = torch.autograd.grad(-loss, weights, retain_graph=True, create_graph=True)

    topn_eigenvalues = []
    eigenvectors = []
    computed_dim = 0
    while computed_dim < top_n:
        eigenvalues = None
        vs = [torch.randn_like(weight) for weight in weights]  # generate random vector
        vs = normalization(vs)  # normalize the vector

        for _ in range(maxIter):
            vs = orthnormal(vs, eigenvectors)
            model.zero_grad()

            Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)
            tmp_eigenvalues = [ torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)]

            vs = normalization(Hvs)

            if eigenvalues == None:
                eigenvalues = tmp_eigenvalues
            else:
                if abs(sum(eigenvalues) - sum(tmp_eigenvalues)) / (abs(sum(eigenvalues)) +
                                                        1e-6) < tol:
                    break
                else:
                    eigenvalues = tmp_eigenvalues
        topn_eigenvalues.append(eigenvalues)
        eigenvectors.append(vs)
        computed_dim += 1

    return topn_eigenvalues, eigenvectors


""" Calculate Hessian Norms: (W-W^)^T (H) (W - W^s)"""
def get_layers(model):
    layers = OrderedDict()
    for name, module in model.named_modules():
        if (type(module) == torch.nn.Linear) and \
            ("LayerNorm" not in name and "embeddings" not in name and "pooler" not in name):
            layers[name] = module
    return layers

def compute_hessians_quantity(model, batch, device="cpu", state_dict = None):
    # Get parameters and gradients of corresponding layer
    batch = prepare_inputs(batch, device)
    outputs = model(**batch)
    loss = outputs.loss

    layers = get_layers(model)
    weights = [module.weight for name, module in layers.items()]
    model.zero_grad()
    gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)
    
    vs = []
    for name, module in layers.items():
        weight = module.weight
        if "pred_head" in name:
            v = weight.detach().clone()
        else:
            v = weight.detach().clone() - state_dict[name+".weight"]
        vs.append(v)

    model.zero_grad()    
    Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)

    layer_hessian_quantities = [torch.sum(Hv*v).cpu().item() for (Hv, v) in zip(Hvs, vs)]
    
    return loss.detach(), np.array(layer_hessian_quantities)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_hessian_traces(model, batch, device, maxIter=200, tol=1e-4):
    batch = prepare_inputs(batch, device)
    outputs = model(**batch)
    loss = outputs.loss

    layers = get_layers(model)
    weights = []
    for name, module in layers.items():
        weights.append(module.weight)
    model.zero_grad()
    gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True)
    
    layer_traces = []
    trace_vhv = []
    trace = 0.
    for _ in range(maxIter):
        vs = [torch.randint_like(weight, high=2) for weight in weights]

        for v in vs:
            v[v==0] = -1
        
        model.zero_grad()
        Hvs = torch.autograd.grad(gradients, weights, grad_outputs=vs, retain_graph=True)
        tmp_layer_traces = np.array([torch.sum(Hv*v).cpu().item() for Hv, v in zip(Hvs, vs)])
        
        layer_traces.append(tmp_layer_traces) 
        trace_vhv.append(sum(tmp_layer_traces))
        
        if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
            break
        else:
            trace = np.mean(trace_vhv)
    return np.mean(np.array(layer_traces), axis=0)