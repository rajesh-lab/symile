import itertools

import torch
import torch.nn.functional as F


def zeroshot_retrieval_logits(r_x, rep_list, logit_scale_exp, loss_fn):
    """
    Computes logits for zeroshot retrieval based on the specified loss function.

    Calculates the logits for predicting the modality r_x using the representations
    in rep_list, and scales the logits by the exponentiated logit scale parameter.

    Args:
        r_x (torch.Tensor): Encoded representations of the modality to predict (num_candidates, d).
        rep_list (list[torch.Tensor]): List of representations for the remaining modalities, each of
                                       size (batch_sz, d) or (d,). This list can can be of any length.
        logit_scale_exp (torch.Tensor): Exponentiated logit scale parameter.
        loss_fn (str): The loss function to use, either "symile" or "clip".

    Returns:
        Tensor: Logits for zeroshot retrieval, of shape (batch_sz, num_candidates).
    """
    if loss_fn == "symile":
        # logits is a (batch_sz, n) matrix where each row i is
        # [ MIP(r_x[i], r_y[i], r_z[0]) ... MIP(r_x[i], r_y[i], r_z[n-1]) ]
        # where MIP is the multilinear inner product.
        product = torch.ones_like(rep_list[0])
        for r in rep_list:
            product *= r

        logits = product @ torch.t(r_x)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
    elif loss_fn == "clip":
        # logits is a (batch_sz, n) matrix where each row i is
        # [ r_x[i]^T r_z[0] + r_z[0]^T r_y[i]   + r_x[i]^T r_y[i] ...
        #   r_x[i]^T r_z[n-1] + r_z[n-1]^T r_y[i] + r_x[i]^T r_y[i] ]
        for i in range(len(rep_list)):
            rep_list[i] = rep_list[i].unsqueeze(0) if rep_list[i].dim() == 1 else rep_list[i] # (batch_sz, d)

        pairwise_sum_with_r_x = torch.zeros_like(rep_list[0] @ torch.t(r_x)) # (batch_sz, num_candidates)
        for r in rep_list:
            pairwise_sum_with_r_x += r @ torch.t(r_x)

        pairwise_sum_without_r_x = torch.zeros((rep_list[0].shape[0], 1), device=rep_list[0].device) # (batch_sz, 1)
        for x, y in itertools.combinations(rep_list, 2):
            pairwise_sum_without_r_x += torch.diagonal(x @ torch.t(y)).unsqueeze(dim=1)

        logits = pairwise_sum_with_r_x + pairwise_sum_without_r_x

    assert logits.dim() == 2, "Logits must be a 2D tensor."

    return logit_scale_exp * logits


########
# clip #
########


def infonce(u, v, logit_scale):
    """
    Computes the CLIP (InfoNCE) loss for a batch of representations.

    Args:
        u, v (torch.Tensor): Representation vectors each of size (batch_sz, d_r).
        logit_scale (torch.Tensor): Learned temperature parameter.
    Returns:
        (torch.Tensor): CLIP (InfoNCE) loss
    """
    logits_u = logit_scale * u @ v.T
    logits_v = logit_scale * v @ u.T

    assert logits_u.shape == logits_v.shape, "Joint embedding spaces must be the same shape."
    labels = torch.arange(logits_u.shape[0]).to(u.device)
    return (F.cross_entropy(logits_u, labels) + F.cross_entropy(logits_v, labels)) / 2.0

def clip(r_a, r_b, r_c, logit_scale, negative_sampling=None):
    """
    Computes the pairwise CLIP loss for a batch of representations.

    Args:
        r_a, r_b, r_c (torch.Tensor): Representation vectors each of size (batch_sz, d_r).
        logit_scale (torch.Tensor): Learned temperature parameter.
        negative_sampling (None): Argument is included for compatibility but is not used in the function.
    Returns:
        (torch.Tensor): Average over the pairwise CLIP (InfoNCE) losses
    """
    loss_ab = infonce(r_a, r_b, logit_scale)
    loss_bc = infonce(r_b, r_c, logit_scale)
    loss_ac = infonce(r_a, r_c, logit_scale)
    return loss_ab + loss_bc + loss_ac


##########
# symile #
##########


def compute_logits_neg_sampling_n(x, y, z):
    """
    Computes the logits for anchor modality x with batch_sz - 1 negatives for
    each positive - or (batch_sz^2 - batch_sz) total negatives.

    If batch_sz is n, then returned logits have size (n, n) with n positive
    multilinear inner products and (n^2 - n) negative multilinear inner products.

    Positive multilinear inner products (MIPs) are along the diagonal of the
    square logits matrix. For example, the second row of `logits` might be:

    [ MIP(x[1], y[3], z[2]) MIP(x[1], y[1], z[1]) MIP(x[1], y[0], z[1]) MIP(x[1], y[2], z[3]) ].

    Notice that only the second element is the positive MIP; all others are negative.
    There is a small chance of a false negative MIP.

    Args:
        x (torch.Tensor): Representation vector of size (batch_sz, d_r).
        y (torch.Tensor): Representation vector of size (batch_sz, d_r).
        z (torch.Tensor): Representation vector of size (batch_sz, d_r).
    Returns:
        logits (torch.Tensor): Logits for x of size (batch_sz, batch_sz).
    """
    # shuffle rows of y and z
    y_shuff = y[torch.randperm(y.shape[0])]
    z_shuff = z[torch.randperm(z.shape[0])]
    logits_x = x @ torch.t(y_shuff * z_shuff) # (batch_sz, batch_sz)
    MIP_of_pos_triples = (x * y * z).sum(axis=1) # (batch_sz)
    # insert positive triples along diagonal of shuffled logits
    return torch.where(torch.eye(n=x.shape[0]).to(x.device) > 0.5, MIP_of_pos_triples, logits_x)


def compute_logits_neg_sampling_n_squared(x, y, z):
    """
    Computes the logits for anchor modality x with batch_sz^2 - 1 negatives for
    each positive.

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
        x (torch.Tensor): Representation vector of size (batch_sz, d_r).
        y (torch.Tensor): Representation vector of size (batch_sz, d_r).
        z (torch.Tensor): Representation vector of size (batch_sz, d_r).
    Returns:
        logits (torch.Tensor): Logits for x of size (batch_sz, batch_sz^2).
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


def symile(r_a, r_b, r_c, logit_scale, negative_sampling):
    """
    Computes the Symile loss for a batch of representations. The final Symile
    loss is an average of the loss terms where each modality is treated as the
    anchor in turn.

    The argument `negative_sampling` can take on one of two values:
        - `n` (for O(n)): draws n - 1 negative samples for each positive
        - `n_squared` (for O(n^2)): draws n^2 - 1 negative samples for each positive

    Args:
        r_a, r_b, r_c (torch.Tensor): Representation vectors each of size (batch_sz, d_r).
        logit_scale (torch.Tensor): Learned temperature parameter.
        negative_sampling (str): Specifies the negative sampling strategy.
                                 Must be either `n` or `n_squared`.

    Returns:
        (torch.Tensor): Average over the losses where each modality is treated
                        as the anchor in turn.
    """
    if negative_sampling == "n":
        logits_a = logit_scale * compute_logits_neg_sampling_n(r_a, r_b, r_c)
        logits_b = logit_scale * compute_logits_neg_sampling_n(r_b, r_a, r_c)
        logits_c = logit_scale * compute_logits_neg_sampling_n(r_c, r_a, r_b)
    elif negative_sampling == "n_squared":
        logits_a = logit_scale * compute_logits_neg_sampling_n_squared(r_a, r_b, r_c)
        logits_b = logit_scale * compute_logits_neg_sampling_n_squared(r_b, r_a, r_c)
        logits_c = logit_scale * compute_logits_neg_sampling_n_squared(r_c, r_a, r_b)
    else:
        raise ValueError("negative_sampling must be either 'n' or 'n_squared'.")

    labels = torch.arange(logits_a.shape[0]).to(r_a.device)
    loss_a = F.cross_entropy(logits_a, labels)
    loss_b = F.cross_entropy(logits_b, labels)
    loss_c = F.cross_entropy(logits_c, labels)
    return (loss_a + loss_b + loss_c) / 3.0