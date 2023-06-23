import torch
import torch.nn.functional as F

####################
# pairwise infonce #
####################
def infonce(u, v, logit_scale):
    logits_u = logit_scale * u @ v.T
    logits_v = logit_scale * v @ u.T

    assert logits_u.shape == logits_v.shape, "Joint embedding spaces must be the same shape."
    labels = torch.arange(logits_u.shape[0])
    return (F.cross_entropy(logits_u, labels) + F.cross_entropy(logits_v, labels)) / 2.0

def pairwise_infonce(r_a, r_b, r_c, logit_scale, normalize=True):
    if normalize:
        r_a = F.normalize(r_a, p=2.0, dim=1)
        r_b = F.normalize(r_b, p=2.0, dim=1)
        r_c = F.normalize(r_c, p=2.0, dim=1)
    loss_ab = infonce(r_a, r_b, logit_scale)
    loss_bc = infonce(r_b, r_c, logit_scale)
    loss_ac = infonce(r_a, r_c, logit_scale)
    return (loss_ab + loss_bc + loss_ac) / 3.0

##########
# symile #
##########
def compute_logits(x, y, z):
    """
    Computes the logits for x, by computing the positive multilinear inner product
    and the negative multilinear inner products for each sample (row) in x.
    Positive multilinear inner products (MIP) are along the diagonal of the logits matrix.
    For example, the second row of `logits` might be:
    [ MIP(x[1], y[3], z[2]) MIP(x[1], y[1], z[1]) MIP(x[1], y[5], z[4]) MIP(x[1], y[2], z[5]) ].
    Notice that only the second element is the positive MIP; all others are negative.
    There is a small chance of a false negative MIP.

    Args:
        x (torch.Tensor): representation vector of size (batch_size, d_r).
        y (torch.Tensor): representation vector of size (batch_size, d_r).
        z (torch.Tensor): representation vector of size (batch_size, d_r).
    Returns:
        logits (torch.Tensor): logits for x of size (batch_size, batch_size).
    """
    # shuffle rows of y and z
    y_shuff = y[torch.randperm(y.shape[0])]
    z_shuff = z[torch.randperm(z.shape[0])]
    def _multilinear_inner_product(x):
        return torch.t(x) @ torch.t(y * z)
    def _multilinear_inner_product_shuffled(x):
        return torch.t(x) @ torch.t(y_shuff * z_shuff)

    logits_ordered = torch.vmap(_multilinear_inner_product)(x)
    logits_x = torch.vmap(_multilinear_inner_product_shuffled)(x)

    # insert positive triples along diagonal of shuffled logits
    logits_x.diagonal().copy_(torch.diag(logits_ordered))
    return logits_x

def symile(r_a, r_b, r_c, logit_scale, normalize):
    if normalize:
        r_a = F.normalize(r_a, p=2.0, dim=1)
        r_b = F.normalize(r_b, p=2.0, dim=1)
        r_c = F.normalize(r_c, p=2.0, dim=1)
    assert r_a.shape == r_b.shape == r_c.shape, "All embeddings must be the same shape."
    logits_a = logit_scale * compute_logits(r_a, r_b, r_c)
    logits_b = logit_scale * compute_logits(r_b, r_a, r_c)
    logits_c = logit_scale * compute_logits(r_c, r_a, r_b)

    labels = torch.arange(logits_a.shape[0])
    loss_a = F.cross_entropy(logits_a, labels)
    loss_b = F.cross_entropy(logits_b, labels)
    loss_c = F.cross_entropy(logits_c, labels)
    return (loss_a + loss_b + loss_c) / 3.0