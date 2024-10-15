# Symile

[Paper] [Datasets] [[Questions]](#questions) [[Citation]](#citation)

Symile is a simple contrastive learning objective that accommodates any number of modalities and allows any model to produce representations for each modality. Symile maintains the simplicity of CLIP while delivering superior performance, even in the case of missing modalities. For a similarity metric, Symile uses the multilinear inner product (MIP), a simple generalization of the dot product to more than two vectors that allows for the simultaneous contrasting of all modalities and enables zero-shot applications such as classification and retrieval.

This repository contains an implementation of the Symile loss and the MIP similarity metric. To reproduce the experiments from the paper (TODO link), follow the instructions in [`experiments/`](https://github.com/rajesh-lab/symile/blob/main/experiments/README.md). TODO: add information on where to find datasets and model checkpoints.

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

https://github.com/openai/CLIP

https://github.com/mlfoundations/open_clip

when you show example, make sure to scale using logit_scale_exp

Here’s an example of how to use the Symile loss in your project:

```
from symile.losses import Symile

logit_scale = torch.tensor(1.0)  # example value
negative_sampling = "n_squared"

# Example of using the Symile loss
loss_fn = Symile(logit_scale, negative_sampling)
rep_list = [...]  # Your batch of representations
loss = loss_fn.forward(rep_list)

from symile.similarity import MipSimilarity

# Initialize the similarity function
similarity_fn = MipSimilarity()

# Example representations
rep_list = [...]  # List of tensors, e.g., [(batch_sz, d), (batch_sz, d)]
r_x = torch.randn(10, 128)  # Candidate modality of size (num_candidates, d)

# Compute similarities
similarity_scores = similarity_fn(rep_list, r_x)
```

### Binary XOR example

We provide an example script that uses Symile to train and test 8 simple linear encoders for the following data generating procedure:

$$\texttt{v_a, v_b, v_c, v_d, v_e, v_f, v_g} \sim \text{Bernoulli}(0.5)$$
$$\texttt{v_h} = \texttt{v_a} \text{ XOR } \texttt{v_b} \text{ XOR } \texttt{v_c} \text{ XOR } \texttt{v_d} \text{ XOR } \texttt{v_e} \text{ XOR } \texttt{v_f} \text{ XOR } \texttt{v_g}$$

The zero-shot classification task is to predict v_a is 0 or 1 given the
remaining variables (v_b, v_c, ..., v_h).

After cloning the repository, from the root directory, first install the necessary dependencies:

```
poetry install --extras "examples"
```

And then run the binary XOR example script:

```
poetry run python examples/binary_xor.py
```


<a name="questions"></a>
## Questions or bugs?
For questions related to the paper, please post on alphaXiv (TODO link) for a prompt response from the authors. For questions related to the code, please open an issue in this repo or email Adriel Saporta (adriel@nyu.edu).

<a name="citation"></a>
## Citation

TODO
