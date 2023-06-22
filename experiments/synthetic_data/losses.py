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
def build_symile_joint_embedding(r_a, r_b, r_c, logit_scale):
    n = r_a.shape[0]
    joint_embed = torch.empty((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                joint_embed[i, j, k] = (r_a[i] * r_b[j] * r_c[k]).sum()
    return logit_scale * joint_embed

def symile_ce_loss(joint_embed, labels):
    logits = joint_embed.flatten(start_dim=1, end_dim=2)
    return F.cross_entropy(logits, labels)

def symile(r_a, r_b, r_c, logit_scale, normalize):
    if normalize:
        r_a = F.normalize(r_a, p=2.0, dim=1)
        r_b = F.normalize(r_b, p=2.0, dim=1)
        r_c = F.normalize(r_c, p=2.0, dim=1)
    assert r_a.shape == r_b.shape == r_c.shape, "All embeddings must be the same shape."
    joint_embed = build_symile_joint_embedding(r_a, r_b, r_c, logit_scale)

    n = joint_embed.shape[0]
    labels = torch.tensor([(i*n)+i for i in range(n)])
    loss_a = symile_ce_loss(joint_embed, labels)
    loss_b = symile_ce_loss(joint_embed.transpose(0, 1), labels)
    loss_c = symile_ce_loss(joint_embed.transpose(0, 2), labels)
    return (loss_a + loss_b + loss_c) / 3.0