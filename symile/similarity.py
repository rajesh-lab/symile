import torch

class MipSimilarity:
    def __init__(self):
        """
        Initializes the MipSimilarity class for computing multilinear inner product similarities.
        """
        pass

    def forward(self, rep_list, r_x):
        """
        Computes similarities between representations in rep_list and the candidate modality r_x.

        Args:
            rep_list (list[torch.Tensor]): List of representations for the query modalities, each of
                                           size (batch_sz, d) or (d,).
            r_x (torch.Tensor): Encoded representations of the candidate modality, size (num_candidates, d).

        Returns:
            torch.Tensor: Similarity scores of size (batch_sz, num_candidates).
        """
        # Compute multilinear inner product similarities
        product = torch.ones_like(rep_list[0])
        for r in rep_list:
            product *= r

        logits = product @ torch.t(r_x)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        return logits

    def __call__(self, rep_list, r_x):
        return self.forward(rep_list, r_x)


def zeroshot_retrieval_logits(r_x, rep_list, logit_scale_exp):
    """
    Computes logits for zeroshot retrieval based on the specified loss function.

    Calculates the logits for predicting the modality r_x using the representations
    in rep_list, and scales the logits by the exponentiated logit scale parameter.

    Args:
        r_x (torch.Tensor): Encoded representations of the modality to predict (num_candidates, d).
        rep_list (list[torch.Tensor]): List of representations for the remaining modalities, each of
                                       size (batch_sz, d) or (d,). This list can can be of any length.
        logit_scale_exp (torch.Tensor): Exponentiated logit scale parameter.

    Returns:
        Tensor: Logits for zeroshot retrieval, of shape (batch_sz, num_candidates).
    """
    # logits is a (batch_sz, n) matrix where each row i is
    # [ MIP(r_x[i], r_y[i], r_z[0]) ... MIP(r_x[i], r_y[i], r_z[n-1]) ]
    # where MIP is the multilinear inner product.
    product = torch.ones_like(rep_list[0])
    for r in rep_list:
        product *= r

    logits = product @ torch.t(r_x)

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    assert logits.dim() == 2, "Logits must be a 2D tensor."

    return logit_scale_exp * logits