import itertools

import torch
import torch.nn.functional as F


class Symile:
    def __init__(self, negative_sampling: str = "n"):
        """
        Initialize the Symile loss function.

        Args:
            negative_sampling (str, optional): Specifies the negative sampling strategy.
                                               Must be either 'n' (for O(n)) or 'n_squared' (for O(n^2)).
                                               Defaults to 'n'.
        """
        self.negative_sampling = negative_sampling

    def compute_logits_n(self, anchor_rep, non_anchor_reps):
        """
        Computes the logits for anchor modality anchor_rep with bsz - 1 negatives for
        each positive - or (bsz^2 - bsz) total negatives. Returned logits have size
        (bsz, bsz) with bsz positive multilinear inner products (MIPs) and (bsz^2 - bsz)
        negative MIPs. Positive MIPs are along the diagonal of the square logits matrix.

        For example, given anchor_rep x and non_anchor_reps y and z, the second row of
        `logits` might be:

        [ MIP(x[1], y[3], z[2]) MIP(x[1], y[1], z[1]) MIP(x[1], y[0], z[1]) MIP(x[1], y[2], z[3]) ].

        Notice that only the second element is the positive MIP; all others are negative.
        There is a small chance of a false negative MIP.

        Args:
            anchor_rep (torch.Tensor): Representation vector for anchor modality (bsz, d).
            non_anchor_reps (list[torch.Tensor]): List of representation tensors for non-anchor
                                                  modalities, each of size (bsz, d). This list
                                                  can contain any number of tensors.
        Returns:
            logits (torch.Tensor): Logits for anchor_rep of size (bsz, bsz).
        """
        # shuffle rows of each tensor in non_anchor_reps and element-wise multiply
        non_anchor_shuff = torch.ones_like(anchor_rep)
        for r in non_anchor_reps:
            # cannot use inplace operations like *= because of autograd
            non_anchor_shuff = non_anchor_shuff * r[torch.randperm(r.shape[0])]

        logits = anchor_rep @ torch.t(non_anchor_shuff) # (bsz, bsz)

        MIP_of_positive_samples = anchor_rep.clone()
        for r in non_anchor_reps:
            # cannot use inplace operations like *= because of autograd
            MIP_of_positive_samples = MIP_of_positive_samples * r
        MIP_of_positive_samples = MIP_of_positive_samples.sum(axis=1) # (bsz)

        # insert positive samples along diagonal of shuffled logits
        return torch.where(torch.eye(n=anchor_rep.shape[0]).to(anchor_rep.device) > 0.5,
                           MIP_of_positive_samples,
                           logits)

    def compute_logits_n_squared(self, anchor_rep, non_anchor_reps):
        """
        Computes the logits for anchor modality anchor_rep with (bsz^len(non_anchor_reps)) - 1
        negatives for each positive. Returned logits have size (bsz, bsz^len(non_anchor_reps))
        with bsz positive multilinear inner products (MIPs) and (bsz^(len(non_anchor_reps)+1) - bsz)
        negative MIPs. Positive MIPs are along the main diagonal of the (non-square) logits matrix.

        For example, given anchor_rep x and non_anchor_reps y and z, and bsz = 4,
        then the second row of `logits` is:

        [ MIP(x[1], y[0], z[0]) MIP(x[1], y[1], z[1]) MIP(x[1], y[2], z[2]) MIP(x[1], y[3], z[3])
          MIP(x[1], y[0], z[3]) MIP(x[1], y[1], z[0]) MIP(x[1], y[2], z[1]) MIP(x[1], y[3], z[2])
          MIP(x[1], y[0], z[2]) MIP(x[1], y[1], z[3]) MIP(x[1], y[2], z[0]) MIP(x[1], y[3], z[1])
          MIP(x[1], y[0], z[1]) MIP(x[1], y[1], z[2]) MIP(x[1], y[2], z[3]) MIP(x[1], y[3], z[0])  ]

        Notice that only the second element is the positive MIP; all others are negative.

        Args:
            anchor_rep (torch.Tensor): Representation vector for anchor modality (bsz, d).
            non_anchor_reps (list[torch.Tensor]): List of representation tensors for non-anchor
                                                  modalities, each of size (bsz, d). This list
                                                  can contain any number of tensors.
        Returns:
            logits (torch.Tensor): Logits for anchor_rep of size (bsz, bsz^len(non_anchor_reps)).
        """
        bsz = anchor_rep.shape[0]
        d = anchor_rep.shape[1]

        # index_combinations is a list of all possible combinations of rows (represented as indices)
        # from each tensor in non_anchor_reps. len(index_combinations) = bsz^len(non_anchor_reps)
        index_combinations = list(itertools.product(range(bsz), repeat=len(non_anchor_reps)))

        # we'll build non_anchor_product so that it has shape (bsz^len(non_anchor_reps), d) where each row
        # is a different element-wise product of a row from each tensor in non_anchor_reps
        non_anchor_product = []

        for indices in index_combinations:
            product = torch.ones(d).to(anchor_rep.device)

            for j, idx in enumerate(indices):
                product = product * non_anchor_reps[j].detach()[idx]

            non_anchor_product.append(product)

        non_anchor_product = torch.stack(non_anchor_product, dim=0)

        logits = anchor_rep @ non_anchor_product.T.detach()

        return logits

    def compute_logits_n_squared_3_modes(self, anchor_rep, non_anchor_reps):
        """
        Computes the logits for anchor modality anchor_rep with bsz^2 - 1 negatives
        for each positive. Returned logits have size (bsz, bsz^2) with bsz positive
        multilinear inner products (MIPs) and (bsz^3 - bsz) negative MIPs.
        Positive MIPs are along the main diagonal of the (non-square) logits matrix.

        This function assumes that len(non_anchor_reps) == 2.

        For example, given anchor_rep x and non_anchor_reps y and z, and bsz = 4,
        then the second row of `logits` is:

        [ MIP(x[1], y[0], z[0]) MIP(x[1], y[1], z[1]) MIP(x[1], y[2], z[2]) MIP(x[1], y[3], z[3])
          MIP(x[1], y[0], z[3]) MIP(x[1], y[1], z[0]) MIP(x[1], y[2], z[1]) MIP(x[1], y[3], z[2])
          MIP(x[1], y[0], z[2]) MIP(x[1], y[1], z[3]) MIP(x[1], y[2], z[0]) MIP(x[1], y[3], z[1])
          MIP(x[1], y[0], z[1]) MIP(x[1], y[1], z[2]) MIP(x[1], y[2], z[3]) MIP(x[1], y[3], z[0])  ]

        Notice that only the second element is the positive MIP; all others are negative.

        Args:
            anchor_rep (torch.Tensor): Representation vector for anchor modality (bsz, d).
            non_anchor_reps (list[torch.Tensor]): List of representation tensors for non-anchor
                                                  modalities, each of size (bsz, d). This list
                                                  must contain exactly two tensors.
        Returns:
            logits (torch.Tensor): Logits for anchor_rep of size (bsz, bsz^2).
        """
        assert len(non_anchor_reps) == 2, "There must be exactly two non-anchor modalities."

        x = anchor_rep
        y, z = non_anchor_reps

        y_z = []
        for i in range(y.shape[0]):
            y_z.append(y * z)
            z = torch.roll(z, shifts=1, dims=0)

        # concatenate elements in y_z so that y_z has shape (bsz^2, d) where each row
        # is a different element-wise product of a row from y and a row from z
        y_z = torch.cat(y_z, 0)

        # return logits with shape (bsz, bsz^2) where each row is the MIP between that
        # row in x and each row from y_z
        logits = x @ y_z.T
        return logits

    def forward(self, representations, logit_scale):
        """
        Computes the Symile loss for a batch of representation vectors.

        Args:
            representations (list[torch.Tensor]): List of representation vectors, each of size (bsz, d).
        Returns:
            (torch.Tensor): Symile loss, which is an average over the losses where each modality is
                            treated as the anchor in turn.
        """
        labels = torch.arange(representations[0].shape[0]).to(representations[0].device)
        losses = []

        for i, r in enumerate(representations):
            if self.negative_sampling == "n":
                logits = logit_scale * self.compute_logits_n(r, [rep for j, rep in enumerate(representations) if i != j])
            elif self.negative_sampling == "n_squared":
                if len(representations) == 3:
                    # three modes allows for a faster implementation
                    logits = logit_scale * self.compute_logits_n_squared_3_modes(r, [rep for j, rep in enumerate(representations) if i != j])
                else:
                    logits = logit_scale * self.compute_logits_n_squared(r, [rep for j, rep in enumerate(representations) if i != j])
            else:
                raise ValueError("Invalid value for negative_sampling. Expected 'n' or 'n_squared'.")

            loss = F.cross_entropy(logits, labels)

            losses.append(loss)

        return sum(losses) / len(losses)

    def __call__(self, representations, logit_scale):
        return self.forward(representations, logit_scale)