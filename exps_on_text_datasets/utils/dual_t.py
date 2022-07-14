import numpy as np
import torch
import torch.nn.functional as F
from utils import MetricTracker, prepare_inputs


def get_transition_matrices(est_loader, model, num_class, device):
    model.eval()
    # est_loader.eval()
    p = []
    T_spadesuit = np.zeros((num_class, num_class))
    with torch.no_grad():
        for i, batch in enumerate(est_loader):
            batch = prepare_inputs(batch, device)
            outputs = model(**batch)
            pred = F.log_softmax(outputs.logits, dim=1)
            # probs = F.softmax(pred, dim=1).cpu().data.numpy()
            probs = torch.exp(pred).cpu().data.numpy()
            _, pred = pred.topk(1, 1, True, True)           
            pred = pred.view(-1).cpu().data
            n_target = batch['labels'].view(-1).cpu().data
            for i in range(len(n_target)): 
                T_spadesuit[int(pred[i])][int(n_target[i])]+=1
            p += probs[:].tolist()  
    T_spadesuit = np.array(T_spadesuit)
    sum_matrix = np.tile(T_spadesuit.sum(axis = 1),(num_class,1)).transpose()
    T_spadesuit = T_spadesuit/sum_matrix
    p = np.array(p)
    T_clubsuit = est_t_matrix(p,filter_outlier=True)
    T_spadesuit = np.nan_to_num(T_spadesuit)
    return T_spadesuit, T_clubsuit

def est_t_matrix(eta_corr, filter_outlier=False):

    # number of classes
    mum_classes = eta_corr.shape[1]
    T = np.empty((mum_classes, mum_classes))

    # find a 'perfect example' for each class
    for i in np.arange(mum_classes):

        if not filter_outlier:
            idx_best = np.argmax(eta_corr[:, i])
        else:
            eta_thresh = np.percentile(eta_corr[:, i], 97, interpolation='higher')
            robust_eta = eta_corr[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
            idx_best = np.argmax(robust_eta)

        for j in np.arange(mum_classes):
            T[i, j] = eta_corr[idx_best, j]

    return T

def compose_T_matrices(T_spadesuit, T_clubsuit):
    dual_t_matrix = np.matmul(T_clubsuit, T_spadesuit)
    return dual_t_matrix