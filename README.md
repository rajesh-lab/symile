# Symile: Simple Model-Agnostic Representation Learning for Unlimited Modalities

<!--- good example repos:
https://github.com/openai/whisper

https://github.com/facebookresearch/segment-anything

https://github.com/stanfordnlp/dspy

https://github.com/openai/CLIP
--->
[`Paper (TODO)`]

Symile is a simple contrastive learning objective that accommodates any number of modalities and allows any model to produce representations for each modality. Symile maintains the simplicity of CLIP while delivering superior performance, even in the case of missing modalities.

This repository contains the datasets and code used to reproduce all experiments in our paper (TODO link). The two original datasets are Symile-M3, an original multilingual dataset of 33M image, text and audio samples, and Symile-MIMIC, a clinical dataset of chest X-rays, electrocardiograms, and laboratory measurements.

This repository contains the datasets and code used to reproduce all experiments in our [paper (TODO link)]. The original datasets include Symile-M3 (TODO link), a multilingual collection of 33 million image, text, and audio samples, and Symile-MIMIC (TODO link), a clinical dataset comprising chest X-rays, electrocardiograms, and laboratory measurements.

We release the weights of all models trained and used for our work TODO.

TODO: keep pyproject.toml as main configuration file, and generate a minimal setup.py file that refers to it, ensuring broader compatibility.

### Table of Contents

This repository contains the official implementation of the Symile loss function. To reproduce the experiments from the paper (TODO link), follow the instructions in [`experiments/`](https://github.com/rajesh-lab/symile/blob/main/experiments/README.md).

- [Installation](#install)
- [Usage](#usage)
- [Questions or bugs?](#questions)
- [License](#license)
- [Citation](#citation)

<a name="install"></a>
## Installation

You can install the Symile package via pip:

```bash
pip install symile
```

#### create environment

TODO: use requirements.txt or pyproject.toml: https://github.com/pypa/packaging.python.org/issues/685#issuecomment-1321616748 and https://venthur.de/2022-12-18-python-packaging.html#:~:text=requirements.,-txt&text=txt%20are%20still%20needed%20if,same%20as%20defined%20in%20pyproject.

Most of the scripts in this project include a command line argument to run with Weights and Biases (W&B) for experiment tracking and visualization. If you'd like to use W&B, please follow the instructions to create an account and install W&B [here](https://docs.wandb.ai/quickstart).

#### activate environment

```
> conda activate symile-env
(symile-env) >
```

#### install project library

From the root directory, run
```
(symile-env) > pip install -e .
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

# Example of using the Symile loss
loss_fn = Symile()
loss = loss_fn(output1, output2, labels)

<a name="questions"></a>
## Questions or bugs?
For questions related to the paper, please post on alphaXiv (TODO link) for a prompt response from the authors. For questions related to the code, please open an issue in this repo or email Adriel Saporta (adriel@nyu.edu).

<a name="license"></a>
## License

TODO

<a name="citation"></a>
## Citation

TODO
