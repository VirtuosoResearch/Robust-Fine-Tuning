"""
Spectral Normalization from https://arxiv.org/abs/1802.05957
"""
import torch
from torch.nn.functional import normalize
from typing import Any, Optional, TypeVar

def reshape_weight_to_matrix(weight: torch.Tensor, dim = 0) -> torch.Tensor:
    weight_mat = weight
    if dim != 0:
        # permute dim to front
        weight_mat = weight_mat.permute(dim,
                                        *[d for d in range(weight_mat.dim()) if d != dim])
    height = weight_mat.size(0)
    return weight_mat.reshape(height, -1)

def compute_spectral_norm(weight, do_power_iteration = True, eps=1e-12, n_power_iterations=10) -> torch.Tensor:
    # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
    #     updated in power iteration **in-place**. This is very important
    #     because in `DataParallel` forward, the vectors (being buffers) are
    #     broadcast from the parallelized module to each module replica,
    #     which is a new module object created on the fly. And each replica
    #     runs its own spectral norm power iteration. So simply assigning
    #     the updated vectors to the module this function runs on will cause
    #     the update to be lost forever. And the next time the parallelized
    #     module is replicated, the same randomly initialized vectors are
    #     broadcast and used!
    #
    #     Therefore, to make the change propagate back, we rely on two
    #     important behaviors (also enforced via tests):
    #       1. `DataParallel` doesn't clone storage if the broadcast tensor
    #          is already on correct device; and it makes sure that the
    #          parallelized module is already on `device[0]`.
    #       2. If the out tensor in `out=` kwarg has correct shape, it will
    #          just fill in the values.
    #     Therefore, since the same power iteration is performed on all
    #     devices, simply updating the tensors in-place will make sure that
    #     the module replica on `device[0]` will update the _u vector on the
    #     parallized module (by shared storage).
    #
    #    However, after we update `u` and `v` in-place, we need to **clone**
    #    them before using them to normalize the weight. This is to support
    #    backproping through two forward passes, e.g., the common pattern in
    #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
    #    complain that variables needed to do backward for the first forward
    #    (i.e., the `u` and `v` vectors) are changed in the second forward.
    with torch.no_grad():
        weight_mat = reshape_weight_to_matrix(weight)

        h, w = weight_mat.size()
        # randomly initialize `u` and `v`
        u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=eps)
        v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=eps)

    if do_power_iteration:
        with torch.no_grad():
            for _ in range(n_power_iterations):
                # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                # are the first left and right singular vectors.
                # This power iteration produces approximations of `u` and `v`.
                v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=eps, out=v)
                u = normalize(torch.mv(weight_mat, v), dim=0, eps=eps, out=u)
            if n_power_iterations > 0:
                # See above on why we need to clone
                u = u.clone(memory_format=torch.contiguous_format)
                v = v.clone(memory_format=torch.contiguous_format)

    sigma = torch.dot(u, torch.mv(weight_mat, v))
    return sigma
