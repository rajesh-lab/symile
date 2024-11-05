# Symile

[Paper](https://arxiv.org/abs/2411.01053) • [Datasets](#datasets) • [Symile vs. CLIP](#symilevclip) • [Questions](#questions) • [Citation](#citation)

Looking to do contrastive pre-training with more than two modalities? Meet Symile!

Symile is a flexible, architecture-agnostic contrastive loss that enables training modality-specific representations for any number of modalities. Symile maintains the simplicity of CLIP while delivering superior performance, even in the case of missing modalities.

For a similarity metric, Symile uses the multilinear inner product (MIP), a simple generalization of the dot product to more than two vectors that allows for the simultaneous contrasting of all modalities and enables zero-shot applications such as classification and retrieval.

## Approach
<img src="/img/symile_summary.png" alt="Symile" width="800"/>
<img src="/img/mip.png" alt="MIP" width="240"/>

To learn more, check out our [paper](https://arxiv.org/abs/2411.01053) (NeurIPS 2024)!

<a name="install"></a>
## Installation

To install the Symile package via pip:

```
pip install symile
```

<a name="usage"></a>
## Usage

Example usage of the Symile loss and MIP similarity metric for three modalities:

```
import torch
import torch.nn.functional as F

from symile import Symile, MIPSimilarity

inputs_a = torch.randn(batch_size, input_dim)
inputs_b = torch.randn(batch_size, input_dim)
inputs_c = torch.randn(batch_size, input_dim)

outputs_a, outputs_b, outputs_c, logit_scale_exp = model(inputs_a, inputs_b, inputs_c)

outputs_a = F.normalize(outputs_a, p=2.0, dim=1)
outputs_b = F.normalize(outputs_b, p=2.0, dim=1)
outputs_c = F.normalize(outputs_c, p=2.0, dim=1)

### train step ###

symile_loss = Symile()
loss = symile_loss([outputs_a, outputs_b, outputs_c], logit_scale_exp)

### evaluation step ###

mip_similarity = MIPSimilarity()

inputs_a_candidates = torch.randn(num_candidates, input_dim)
outputs_a_candidates = model.encoder_a(inputs_a_candidates)
outputs_a_candidates = F.normalize(outputs_a_candidates, p=2.0, dim=1)

similarity_scores = mip_similarity(outputs_a_candidates, [outputs_b, outputs_c])
similarity_scores = logit_scale_exp * similarity_scores
```

## Example

We provide a very simple example script that uses the Symile loss and the MIP similarity metric to train and test 8 linear encoders for the following data generating procedure:

**a**, **b**, **c**, **d**, **e**, **f**, **g** $\sim$ Bernoulli(0.5)

**h** $=$ **a** $\text{ XOR }$ **b** $\text{ XOR }$ **c** $\text{ XOR }$ **d** $\text{ XOR }$ **e** $\text{ XOR }$ **f** $\text{ XOR }$ **g**

The zero-shot classification task is to predict whether **a** is 0 or 1 given the remaining variables **b**, **c**, **d**, **e**, **f**, **g**, **h**.

After cloning the repository, first install the necessary dependencies from the root directory and then run the script:

```
> poetry install --with examples
> poetry run python examples/binary_xor.py
```

## Negative sampling

Symile learns by contrasting positive samples with negative samples. Like CLIP, Symile constructs negatives for each positive by using other samples within the batch. Let's say you have a batch of 4 samples, consisting of three modalities `A`, `B`, and `C`:
```
A1 B1 C1
A2 B2 C2
A3 B3 C3
A4 B4 C4
```
Each of the above triples is a positive sample. How do we construct negatives? Symile offers two strategies: $O(N)$ and $O(N^2)$. The $O(N)$ strategy is the default as it provides a good balance between efficiency and effectiveness for most use cases. For smaller datasets, the $O(N^2)$ strategy can help prevent overfitting by exposing your model to more negative examples.

### 1. $O(N)$: fast and memory efficient

This approach randomly shuffles the non-anchor modalities to create $N-1$ negatives per positive. For example, if `A1` is our anchor, we might get:
```
Positive:  A1-B1-C1
Negatives: A1-B3-C4
           A1-B4-C2
           A1-B2-C3
```
To use this approach, you can either initialize `Symile()` with no arguments, or explicitly set the `negative_sampling` argument:
```
symile_loss = Symile()
# or
symile_loss = Symile(negative_sampling="n")
```
### 2. $O(N^2)$: maximum coverage

This approach creates all possible combinations of non-anchor modalities, creating $N^2 - 1$ negatives per positive (the cube in the pre-training figure above illustrates this approach). Using `A1` as our anchor again:
```
Positive:  A1-B1-C1
Negatives:           A1-B1-C2, A1-B1-C3, A1-B1-C4
           A1-B2-C1, A1-B2-C2, A1-B2-C3, A1-B2-C4
           A1-B3-C1, A1-B3-C2, A1-B3-C3, A1-B3-C4
           A1-B4-C1, A1-B4-C2, A1-B4-C3, A1-B4-C4
```
To use the $O(N^2)$ approach:
```
symile_loss = Symile(negative_sampling="n_squared")
```

## Missing data

What if some samples in your dataset don’t contain all modalities? For instance, a patient may be missing lab results, or a social media post might not include an image. **Symile can be easily adapted to handle missing modalities** by passing as inputs to the model both the data (using any placeholder value for missing modalities) and binary indicators that signal which modalities are present for each sample. This approach lets Symile model the relationships between whichever modalities are present in each sample.

We provide a simple script demonstrating how to train Symile with missing modalities. The data is generated as follows:

**a**, **b** $\sim$ Bernoulli(0.5) $\qquad$ **c** $=$ **a** $\text{ XOR }$ **b**

The zero-shot classification task is to predict whether **a** is 0 or 1 given the remaining variables **b**, **c**. To simulate missingness in the training and validation sets, values in **a**, **b**, and **c** are randomly set to 0.5 with probability `args.missingness_prob`. The vectors **a**, **b**, **c** and their missingness indicators are then passed to the encoders. To run the script:

```
> poetry install --with examples
> poetry run python examples/binary_xor_missing.py
```

Note that instead of using binary indicators, you could also use any out-of-support placeholder to represent missing data (provided your model is expressive enough). Binary indicators provide a simple way to ensure missing data is out-of-support, but other approaches work, too. For example, with text data, you could use a special token that's outside of your model's vocabulary (e.g., `[MISSING]`), as we did in our paper's experiments.

<a name="datasets"></a>
## Datasets

As part of this research, we release two novel multimodal datasets:
* **Symile-M3:** a multilingual collection of 33 million image, text, and audio samples.
* **Symile-MIMIC:** a clinical dataset of chest X-rays, electrocardiograms, and laboratory measurements.

> Note: Both datasets are scheduled for public release. Follow this repository for updates.

To reproduce the experiments from our paper using these datasets, navigate to the `experiments/` directory and follow the step-by-step instructions in the dedicated README.

<a name="symilevclip"></a>
## Symile vs. CLIP

The Symile loss targets _total correlation_, which is the higher-order generalization of mutual information to any number of random variables. Total correlation can be decomposed into a summation of mutual information terms. For example, in the case of three random variables,

<img src="/img/tc_equation.png" alt="Total correlation equation" width="675"/>

While, like many contrastive approaches, CLIP was designed to capture the shared information between modalities, the above equation indicates that when there are more than two modalities, the scope of what to capture should extend beyond pairwise information to include conditional interactions. Because it targets total correlation, **Symile captures _strictly more_ information than CLIP, guaranteeing performance that matches or surpasses CLIP!**
<p>
<img src="/img/tc_illustration.png" alt="Total correlation illustration" align="left" style="margin-right: 10px; margin-bottom: 20px; width: 330px;"/>
Most real-world applications will exhibit a combination of both pairwise and higher-order information. For example, in order to diagnose acute pancreatitis, one might consider a patient’s clinical history of abdominal pain, elevated levels of digestive enzymes, and imaging results consistent with inflammation. While each of these modalities would provide useful information about the likelihood of pancreatitis (i.e., pairwise information between the modality and the diagnosis is non-zero), none of them alone would be diagnostic of the condition.
</p>

**Bottom line:** if you're looking to do contrastive pre-training with more than two modalities, use Symile!

<a name="questions"></a>
## Questions?
We welcome all questions and feedback! Here's how to reach us:
- **Paper:** Join the discussion on [alphaXiv](https://www.alphaxiv.org/abs/2411.01053).
- **Code:** Feel free to open an issue in this repository.
- **Contact:** Shoot Adriel an email at `adriel@nyu.edu`.

Please don't hesitate to reach out—your questions help make this project better for everyone! 🚀

<a name="citation"></a>
## Citation

```
@inproceedings{saporta2024symile,
  title = {Contrasting with Symile: Simple Model-Agnostic Representation Learning for Unlimited Modalities}
  author = {Saporta, Adriel and Puli, Aahlad and Goldstein, Mark and Ranganath, Rajesh}
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2024}
}
```
