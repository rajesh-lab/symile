import torch.nn.functional as F


def l2_normalize(vectors):
    """
    L2 normalize a list of 2D vectors.

    Args:
        vectors (list): list of 2D torch.Tensor vectors.
    Returns:
        list of same 2D torch.Tensor vectors, normalized.
    """
    return [F.normalize(v, p=2.0, dim=1) for v in vectors]