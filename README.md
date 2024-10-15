# Symile

[Paper] [Datasets] [[Questions]](#questions) [[Citation]](#citation)

Symile is a simple contrastive learning objective that accommodates any number of modalities and allows any model to produce representations for each modality. Symile maintains the simplicity of CLIP while delivering superior performance, even in the case of missing modalities. For a similarity metric, Symile uses the multilinear inner product (MIP), a simple generalization of the dot product to more than two vectors that allows for the simultaneous contrasting of all modalities and enables zero-shot applications such as classification and retrieval.

This repository contains an implementation of the Symile loss and the MIP similarity metric. To reproduce the experiments from the paper (TODO link), follow the instructions in `experiments/` [directory](https://github.com/rajesh-lab/symile/blob/main/experiments/README.md). TODO: add information on where to find datasets and model checkpoints.

The original datasets include Symile-M3 (TODO link), a multilingual collection of 33 million image, text, and audio samples, and Symile-MIMIC (TODO link), a clinical dataset comprising chest X-rays, electrocardiograms, and laboratory measurements.

## Approach

TODO: add figure.

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

from symile.loss import Symile
from symile.similarity import MIPSimilarity

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

We provide a very simple example script that uses the Symile loss and the MIP similarity metric to train and test 8 simple linear encoders for the following data generating procedure:

**a**, **b**, **c**, **d**, **e**, **f**, **g** $\sim$ Bernoulli(0.5)

**h** $=$ **a** $\text{ XOR }$ **b** $\text{ XOR }$ **c** $\text{ XOR }$ **d** $\text{ XOR }$ **e** $\text{ XOR }$ **f** $\text{ XOR }$ **g**

The zero-shot classification task is to predict whether **a** is 0 or 1 given the remaining variables **b**, **c**, **d**, **e**, **f**, **g**, **h**.

After cloning the repository, from the root directory, first install the necessary dependencies:

```
poetry install --with examples
```

And then run the binary XOR example script:

```
poetry run python examples/binary_xor.py
```

<a name="questions"></a>
## Questions or bugs?
For questions related to the paper, please post on alphaXiv (TODO link) for a prompt response from the authors. For questions related to the code, please open an issue in this repo or email Adriel (adriel@nyu.edu).

<a name="citation"></a>
## Citation

TODO
