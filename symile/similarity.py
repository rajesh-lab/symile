import torch

class MIPSimilarity:
    def __init__(self):
        """
        Initializes class for computing multilinear inner product (MIP) similarities.
        """
        pass

    def forward(self, candidates, query_reps):
        """
        Returns the similarity scores for predicting the candidate modality using the query modalities.

        Specifically, the i-th row of the returned similarity score matrix contains the multilinear
        inner products (MIPs) between the i-th query modalities and all candidates. For example,
        given the candidate modality x and the query modalities y and z, the i-th row in the returned
        similarity score matrix is:

        [ MIP(x[i], y[i], z[0]), MIP(x[i], y[i], z[1]), ..., MIP(x[i], y[i], z[num_candidates-1]) ]

        Args:
            candidates (torch.Tensor): Representations for the candidate modality (num_candidates, d).
            query_reps (list[torch.Tensor]): List of tensors, where each tensor corresponds to the
                                             representations of a single query modality. Each tensor in the
                                             list has shape (bsz, d) or (d,), where bsz is the number of
                                             queries. The list can contain any number of modalities.

        Returns:
            torch.Tensor: Similarity scores of size (bsz, num_candidates), where each score corresponds to
                          the similarity between the query modalities and the candidate modality for each
                          query in the batch.
        """
        query_product = torch.ones_like(query_reps[0])
        for r in query_reps:
            query_product *= r

        similarity_scores = query_product @ torch.t(candidates)

        if similarity_scores.dim() == 1:
            similarity_scores = similarity_scores.unsqueeze(0)

        assert similarity_scores.dim() == 2, "similarity_scores must be a 2D tensor."

        return similarity_scores

    def __call__(self, candidates, query_reps):
        return self.forward(candidates, query_reps)