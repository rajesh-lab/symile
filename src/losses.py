import torch
import torch.nn.functional as F


####################
# pairwise infonce #
####################
def infonce(u, v, logit_scale):
    """
    Computes the InfoNCE loss for a batch of representations.

    Args:
        u, v (torch.Tensor): representation vectors each of size (batch_sz, d_r).
        logit_scale (torch.Tensor): temperature parameter as a log-parameterized
                                    multiplicative scalar (see CLIP).
    Returns:
        (torch.Tensor): InfoNCE loss
    """
    logits_u = logit_scale * u @ v.T
    logits_v = logit_scale * v @ u.T

    assert logits_u.shape == logits_v.shape, "Joint embedding spaces must be the same shape."
    labels = torch.arange(logits_u.shape[0]).to(u.device)
    return (F.cross_entropy(logits_u, labels) + F.cross_entropy(logits_v, labels)) / 2.0

def pairwise_infonce(r_a, r_b, r_c, logit_scale):
    """
    Computes the pairwise InfoNCE loss for a batch of representations.

    Args:
        r_a, r_b, r_c (torch.Tensor): representation vectors each of size (batch_sz, d_r).
    Returns:
        (torch.Tensor): average over the pairwise InfoNCE losses
    """
    loss_ab = infonce(r_a, r_b, logit_scale)
    loss_bc = infonce(r_b, r_c, logit_scale)
    loss_ac = infonce(r_a, r_c, logit_scale)
    return loss_ab + loss_bc + loss_ac

##########
# symile #
##########
def compute_logits_efficient(x, y, z):
    """
    Computes the logits for x with only (batch_size^2 - batch_size) negatives.

    If batch size is n, then returned logits have size (n, n) with n positive
    multilinear inner products and (n^2 - n) negative multilinear inner products.

    Positive multilinear inner products (MIP) are along the diagonal of the
    square logits matrix. For example, the second row of `logits` might be:

    [ MIP(x[1], y[3], z[2]) MIP(x[1], y[1], z[1]) MIP(x[1], y[0], z[1]) MIP(x[1], y[2], z[3]) ].

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
    logits_x = x @ torch.t(y_shuff * z_shuff) # (batch_sz, batch_sz)
    MIP_of_pos_triples = (x * y * z).sum(axis=1) # (batch_sz)
    # insert positive triples along diagonal of shuffled logits
    return torch.where(torch.eye(n=x.shape[0]).to(x.device) > 0.5, MIP_of_pos_triples, logits_x)


def compute_logits(x, y, z):
    """
    Computes the logits for x.

    If batch size is n, then returned logits have size (n, n^2) with n positive
    multilinear inner products and (n^3 - n) negative multilinear inner products.

    Positive multilinear inner products (MIP) are along the main diagonal of the
    (non-square) logits matrix. For example, if n = 4, then the second row of
    `logits` is:

    [ MIP(x[1], y[0], z[0]) MIP(x[1], y[1], z[1]) MIP(x[1], y[2], z[2]) MIP(x[1], y[3], z[3])
      MIP(x[1], y[0], z[3]) MIP(x[1], y[1], z[0]) MIP(x[1], y[2], z[1]) MIP(x[1], y[3], z[2])
      MIP(x[1], y[0], z[2]) MIP(x[1], y[1], z[3]) MIP(x[1], y[2], z[0]) MIP(x[1], y[3], z[1])
      MIP(x[1], y[0], z[1]) MIP(x[1], y[1], z[2]) MIP(x[1], y[2], z[3]) MIP(x[1], y[3], z[0])  ]

    Notice that only the second element is the positive MIP; all others are negative.

    Args:
        x (torch.Tensor): representation vector of size (batch_size, d_r).
        y (torch.Tensor): representation vector of size (batch_size, d_r).
        z (torch.Tensor): representation vector of size (batch_size, d_r).
    Returns:
        logits (torch.Tensor): logits for x of size (batch_size, batch_size^2).
    """
    y_z = []
    for i in range(y.shape[0]):
        y_z.append(y * z)
        z = torch.roll(z, shifts=1, dims=0)

    # concatenate elements in y_z so that y_z has shape (n^2, d) where each row
    # is a different element-wise product of a row from y and a row from z
    y_z = torch.cat(y_z, 0)

    # return logits with shape (n, n^2) where each row is the multilinear inner
    # product between that row in x and each row from y_z
    logits = x @ y_z.T
    return logits


def symile(r_a, r_b, r_c, logit_scale):
    logits_a = logit_scale * compute_logits(r_a, r_b, r_c)
    logits_b = logit_scale * compute_logits(r_b, r_a, r_c)
    logits_c = logit_scale * compute_logits(r_c, r_a, r_b)

    labels = torch.arange(logits_a.shape[0]).to(r_a.device)
    loss_a = F.cross_entropy(logits_a, labels)
    loss_b = F.cross_entropy(logits_b, labels)
    loss_c = F.cross_entropy(logits_c, labels)
    return (loss_a + loss_b + loss_c) / 3.0