# Symile

The `experiments/` directory contains all code required to reproduce the three sets of experiments presented in our [paper](https://arxiv.org/abs/2411.01053):
- [Synthetic data](#binary_xor)
- [Symile-M3](#symile_m3): a multilingual dataset of 33 million (audio, image, text) samples.
- [Symile-MIMIC](#symile_mimic): a clinical dataset of chest X-rays, electrocardiograms, and laboratory measurements.

<a name="setup"></a>
## Setup

#### download datasets

The scripts in this repository assume that you've downloaded the Symile-M3 and Symile-MIMIC datasets, preprocessed the dataset splits, and saved the resulting tensors in split-specific directories, as detailed [here for Symile-M3](https://github.com/rajesh-lab/symile/blob/main/experiments/data_processing/symile_m3/README.md) and [here for Symile-MIMIC](https://github.com/rajesh-lab/symile/blob/main/experiments/data_processing/symile_mimic/README.md).

#### activate environment

To start, make sure you're in the correct directory:
```
> cd experiments
```

##### conda

```
> conda env create -f env/environment.yml
> conda activate symile-env
(symile-env) >
```

##### pip

```
> python -m venv venv
> source venv/bin/activate
(venv) > pip install -r env/requirements.txt
```

#### install experiments package

```
(symile-env) > pip install -e .
```

#### w&b

Most of the scripts in this project include a command line argument to run with Weights and Biases (W&B) for experiment tracking and visualization. If you'd like to use W&B, follow the instructions to create an account and install W&B [here](https://docs.wandb.ai/quickstart).

<a name="pretrain"></a>
## Pre-training

The following command-line arguments are common to all three sets of experiments and can be specified when running `main.py`.

| Flag                          | Description                                                   | Type                | Choices                          | Default |
|-------------------------------|---------------------------------------------------------------|---------------------|----------------------------------|---------------|
| `--experiment`                | Experiment identifier                                         | str                 | `binary_xor`, `symile_m3`, `symile_mimic` |               |
| `--batch_sz_train`            | Batch size for training                                       | int                 |                                  |               |
| `--batch_sz_val`              | Batch size for validation                                     | int                 |                                  |               |
| `--batch_sz_test`             | Batch size for testing                                        | int                 |                                  |               |
| `--d`                         | Shared dimensionality for all learned representations         | int                 |                                  |               |
| `--data_dir`                  | Directory with dataset csv files                              | Path                |                                  |         |
| `--epochs`                    | Number of training epochs                                     | int                 |                                  |               |
| `--check_val_every_n_epoch`   | Frequency of validation checks (in epochs)                    | int                 |                                  |               |
| `--drop_last`                 | Drop the last incomplete training batch if the training set is not divisible by batch size | bool | `True`, `False`                  |               |
| `--lr`                        | Learning rate                                                 | float               |                                  |               |
| `--weight_decay`                        | Weight decay coefficient used by AdamW optimizer | float               |                                  | 0.01 |
| `--logit_scale_init`          | Initial value for the logit scale, which is the temperature parameter $\tau$ is directly optimized during training as a multiplicative scalar to avoid having to tune it as a hyperparameter | float | | |
| `--negative_sampling`         | Negative sampling strategy: $O(N)$ or $O(N^2)$                | str                 | `n`, `n_squared`                 |               |
| `--loss_fn`                   | Loss function to use                                          | str                 | `symile`, `clip`        |  `symile`             |
| `--ckpt_path`            | Path of the checkpoint from which training is resumed | str                 |         | `None` |
| `--ckpt_save_dir`             | Directory to save checkpoints                                 | str                 |         |               |

The following arguments are helpful for debugging and are set with default values non-debugging use:

| Flag                          | Description                                                   | Type                | Choices                          | Default |
|-------------------------------|---------------------------------------------------------------|---------------------|----------------------------------|---------------|
| `--limit_train_batches`       | Fraction of training batches to use (e.g. set to 0.1 to check 10% of dataset) | float | Any float between 0.0 and 1.0  | 1.0 |
| `--limit_val_batches`         | Fraction of validation batches to use (e.g. set to 0.1 to check 10% of dataset) | float | Any float between 0.0 and 1.0| 1.0 |
| `--freeze_logit_scale`        | Whether to freeze the logit scale                             | bool                | `True`, `False`                  | `False`       |
| `--use_seed`                  | Use a seed for reproducibility                                | bool                | `True`, `False`                  | `False`       |
| `--seed`                      | Random seed for reproducibility                               | int                 |                                  | 0             |
| `--wandb`                     | Enable Weights and Biases for logging                         | bool                | `True`, `False`                  | `False`       |
| `--wandb_run_id`                     | Use if loading from checkpoint and using WandbLogger   | str                |                   |       |

<a name="binary_xor"></a>
## Synthetic data experiments

In this section, we reproduce the synthetic data experiment from Section 5.1 of our [paper](https://arxiv.org/abs/2411.01053) in which the dataset is drawn according to the following sampling procedure:

$$a_j, b_j \sim \text{Bernoulli}(0.5), \quad i \sim \text{Bernoulli}(\hat{p}), \quad c_j = (a_j \text{ XOR } b_j)^i \cdot a_j^{(1-i)}$$
$$\mathbf{a} = [a_1,\dots, a_d], \quad \mathbf{b} = [b_1,\dots, b_d], \quad \mathbf{c} = [c_1,\dots, c_d].$$

The following command runs the experiment for values of $\hat{p}$ in $`\{0.0, 0.1,0.2,\dots,1.0\}`$:

```
(symile-env) > python main.py --experiment binary_xor [FLAGS]
```

In addition to the [common pre-training command-line arguments](#pretrain), this command takes the following experiment-specific flags:

| Flag        | Description                               | Type   | Choices           | Default |
|-------------|-------------------------------------------|--------|-------------------|---------|
| `--train_n` | Number of training samples to draw        | int    |  |    |
| `--val_n`   | Number of validation samples to draw      | int    |  |     |
| `--test_n`  | Number of test samples to draw            | int    |  |     |
| `--d_v`     | Dimensionality of the input vectors $\mathbf{a}$, $\mathbf{b}$, and $\mathbf{c}$ | int |  |  |
| `--bootstrap`                 | Whether to bootstrap test results                            | bool                | `True`, `False`                  | `False` |
| `--bootstrap_n`               | Number of bootstrap samples                                  | int                 |                                  | 10    |

### Calculate information terms

We also include the code to track the changing information dynamics between the variables $\mathbf{a}$, $\mathbf{b}$, and $\mathbf{c}$ as $\hat{p}$ moves from 0 to 1 (Figure 3 in the [paper](https://arxiv.org/abs/2411.01053).
Specifically, the following command calculates $\mathbf{I}(\mathbf{a};\mathbf{c})$, $\mathbf{I}(\mathbf{b};\mathbf{c})$, $`\mathbf{I}(\mathbf{a};\mathbf{b}\,|\,\mathbf{c})`$, $`\mathbf{I}(\mathbf{c};\mathbf{b}\,|\,\mathbf{a})`$, and $\mathbf{TC}(\mathbf{a},\mathbf{b},\mathbf{c})$ for each $\hat{p}$ in $`\{0.0, 0.1,0.2,\dots,1.0\}`$:

```
(symile-env) > python ./data_processing/binary_xor/informations.py --d_v <input_vector_dim> --save_dir <path/to/save_dir>
```

Note that running this script for `d_v = 5` takes about 1.5 hours.

<a name="symile_m3"></a>
## Symile-M3 experiments
<img src="/img/symile_m3.png" alt="Symile-M3" width="800"/>

In this section, we reproduce the Symile-M3 experiments from Section 5.2 of our [paper](https://arxiv.org/abs/2411.01053). These scripts assume that you've downloaded the Symile-M3 dataset, preprocessed the dataset splits, and saved the resulting tensors in split-specific directories, as detailed [here](https://github.com/rajesh-lab/symile/blob/main/experiments/data_processing/symile_m3/README.md).

The following command runs pretraining on Symile-M3:

```
(symile-env) > python main.py --experiment symile_m3 [FLAGS]
```

In addition to the [common pre-training command-line arguments](#pretrain), this command takes the following experiment-specific flags:

| Flag                    | Description                                         | Type   | Choices                        | Default                                                  |
|-------------------------|-----------------------------------------------------|--------|--------------------------------|----------------------------------------------------------|
| `--audio_model_id`      | Hugging Face model id for audio encoder             | str    |       |  |
| `--image_model_id`      | Hugging Face model id for image encoder             | str    | |  |
| `--text_model_id`       | Hugging Face model id for text encoder              | str    |             |  |
| `--train_csv`           | Filename for train CSV                              | Path   | Any valid file path            | `train.csv`                                              |
| `--val_csv`             | Filename for val CSV                                | Path   | Any valid file path            | `val.csv`                                                |
| `--test_csv`            | Filename for test CSV                               | Path   | Any valid file path            | `test.csv`                                               |
| `--num_langs`           | Number of languages                                 | int    | 2, 5, or 10           |                                                      |
| `--translations_path`      | Path to JSON file with ImageNet class names, synset ids, and translations                | str    |             |  |
| `--missingness`               | Whether to train with missingness                             | bool                | `True`, `False`                  | `False`        |
| `--missingness_prob`          | Probability with which a given modality is missing            | float  | Any float between 0.0 and 1.0  |  |
| `--text_embedding`      | Specifies whether to use text encoder BOS or EOS embedding as input to projection head | str    | `eos`, `bos`         | `eos`     |
| `--metadata_filename`   | Path to JSON file with metadata for all encoders (generated by `symile/experiments/data_processing/symile_m3/save_representations.py`    | Path   | Any valid file path            | `metadata.json`                                          |

The `metadata_filename` is generated by `save_representations.py` as described [here](https://github.com/rajesh-lab/symile/blob/main/experiments/data_processing/symile_m3#preprocess-and-save-dataset-tensors). Missingness is described in more detail [here](https://github.com/rajesh-lab/symile/tree/main/experiments/data_processing/symile_m3#missingness).

### Evaluation

The following command will evaluate a given model (i.e. checkpoint) on Symile-M3:

```
(symile-env) > python test.py \
    --experiment symile_m3 \
    --batch_sz_test 256 \
    --bootstrap True \
    --bootstrap_n 10 \
    --data_dir /path/to/dataset \
    --description "Symile-M3 experiment" \
    --ckpt_path /path/to/checkpoint.ckpt \
    --save_dir /path/to/save_results \
    --seed 0 \
    --use_seed True \
    --num_langs 10 (must be 2, 5, or 10)
```

<a name="symile_mimic"></a>
## Symile-MIMIC experiments
<img src="/img/symile_mimic.png" alt="Symile-MIMIC" width="400"/>

In this section, we reproduce the Symile-MIMIC experiments from Section 5.3 of our [paper](https://arxiv.org/abs/2411.01053). These scripts assume that you've downloaded the Symile-MIMIC dataset, preprocessed the dataset splits, and saved the resulting tensors in split-specific directories, as detailed [here](https://github.com/rajesh-lab/symile/blob/main/experiments/data_processing/symile_mimic/README.md).

The following command runs pretraining on Symile-MIMIC:

```
(symile-env) > python main.py --experiment symile_mimic [FLAGS]
```

In addition to the [common pre-training command-line arguments](#pretrain), this command takes the following experiment-specific flags:

| Flag                    | Description                                         | Type   | Choices                        | Default                                                  |
|-------------------------|-------------------------------------------------------------|--------|----------------------|-----------|
| `--pretrained`          | Whether to use pretrained encoders for CXR and ECG          | bool   | `True`, `False`      | `False`   |

If `pretrained` is `True`, the CXR encoder (ResNet-50) is initialized with ImageNet (`IMAGENET1K_V2`) weights, and the ECG encoder (ResNet-18) is initialized with ImageNet (`IMAGENET1K_V1`) weights.

### Evaluation

The following command will evaluate a given model (i.e. checkpoint) on Symile-MIMIC:

```
(symile-env) > python test.py \
    --experiment symile_mimic \
    --batch_sz_test 256 \
    --bootstrap True \
    --bootstrap_n 10 \
    --data_dir /path/to/dataset \
    --description "Symile-MIMIC experiment" \
    --ckpt_path /path/to/checkpoint.ckpt \
    --save_dir /path/to/save_results \
    --seed 0 \
    --use_seed True
```
