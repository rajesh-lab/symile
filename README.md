# Symile

<!--- good example repos:
https://github.com/openai/whisper

https://github.com/facebookresearch/segment-anything
--->
[`Paper (TODO)`]

Symile is a simple contrastive learning objective that accommodates any number of modalities and allows any model to produce representations for each modality. Symile maintains the simplicity of CLIP while delivering superior performance, even in the case of missing modalities.

This repository contains the datasets and code used to reproduce all experiments in our paper (TODO link). The two original datasets are Symile-M3, an original multilingual dataset of 33M image, text and audio samples, and Symile-MIMIC, a clinical dataset of chest X-rays, electrocardiograms, and laboratory measurements.

This repository contains the datasets and code used to reproduce all experiments in our [paper (TODO link)]. The original datasets include Symile-M3 (TODO link), a multilingual collection of 33 million image, text, and audio samples, and Symile-MIMIC (TODO link), a clinical dataset comprising chest X-rays, electrocardiograms, and laboratory measurements.

We release the weights of all models trained and used for our work TODO.

### Table of Contents
- [Set up environment](#environment)
- [Pre-training command-line arguments](#pretrain_args)
- [Binary XOR experiments](#binary_xor)
- [Symile-M3 experiments](#symile_m3)
- [Symile-MIMIC experiments](#cxr_prediction)
- [Questions?](#questions)
- [License](#license)
- [Citation](#citation)

<a name="environment"></a>
## Set up environment

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

<a name="pretrain_args"></a>
## Pre-training command-line arguments

The following command-line arguments are common to all three sets of experiments (Binary XOR, Symile-M3, and Symile-MIMIC) and can be specified when running `main.py`.

TODO: make sure wandb set to false by default, and set symile to default loss function, and make sure use_seed is False by default. also update bash script examples so people have default... or include a txt with examples. maybe make save ckpts the default be current directory

| Flag                          | Description                                                   | Type                | Choices                          | Default |
|-------------------------------|---------------------------------------------------------------|---------------------|----------------------------------|---------------|
| `--experiment`                | Experiment identifier                                         | str                 | `binary_xor`, `symile_m3`, `symile_mimic` |               |
| `--batch_sz_train`            | Batch size for training                                       | int                 |                                  |               |
| `--batch_sz_val`              | Batch size for validation                                     | int                 |                                  |               |
| `--batch_sz_test`             | Batch size for testing                                        | int                 |                                  |               |
| `--d`                         | Shared dimensionality for all learned representations         | int                 |                                  |               |
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

<a name="binary_xor"></a>
## Binary XOR experiments

In this section, we reproduce the binary XOR experiment from Section 5.1 of the paper (TODO link) in which a synthetic dataset is drawn according to the following sampling procedure:
$$a_j, b_j \sim \text{Bernoulli}(0.5), \quad i \sim \text{Bernoulli}(\hat{p}), \quad c_j = (a_j \text{ XOR } b_j)^i \cdot a_j^{(1-i)}$$
$$\mathbf{a} = [a_1,\dots, a_d], \quad \mathbf{b} = [b_1,\dots, b_d], \quad \mathbf{c} = [c_1,\dots, c_d].$$

The following command runs the binary XOR experiment for values of $\hat{p}$ in $`\{0.0, 0.1,0.2,\dots,1.0\}`$:

```
(symile-env) > python main.py --experiment binary_xor [FLAGS]
```

In addition to the [common pre-training command-line arguments](#pretrain_args), this command takes the following experiment-specific flags:

| Flag        | Description                               | Type   | Choices           | Default |
|-------------|-------------------------------------------|--------|-------------------|---------|
| `--train_n` | Number of training samples to draw        | int    |  |    |
| `--val_n`   | Number of validation samples to draw      | int    |  |     |
| `--test_n`  | Number of test samples to draw            | int    |  |     |
| `--d_v`     | Dimensionality of the input vectors $\mathbf{a}$, $\mathbf{b}$, and $\mathbf{c}$ | int |  |  |

### Calculate information terms

We also include the code to track the changing information dynamics between the variables $\mathbf{a}$, $\mathbf{b}$, and $\mathbf{c}$ as $\hat{p}$ moves from 0 to 1 (Figure 3b in the paper (TODO link)).
Specifically, the following command calculates $\mathbf{I}(\mathbf{a};\mathbf{c})$, $\mathbf{I}(\mathbf{b};\mathbf{c})$, $`\mathbf{I}(\mathbf{a};\mathbf{b}\,|\,\mathbf{c})`$, $`\mathbf{I}(\mathbf{c};\mathbf{b}\,|\,\mathbf{a})`$, and $\mathbf{TC}(\mathbf{a},\mathbf{b},\mathbf{c})$ for each $\hat{p}$ in $`\{0.0, 0.1,0.2,\dots,1.0\}`$:

```
(symile-env) > python ./data_processing/binary_xor/informations.py --d_v <input_vector_dim> --save_dir <path/to/save_dir>
```

Note that running this script for `d_v = 5` takes about 1.5 hours.

<a name="symile_m3"></a>
## Symile-M3 experiments

![symile_m3](/img/symile_m3.png)

In this section, we describe how to access Symile-M3, a new multilingual dataset comprising 33 million (audio, image, text) samples. We also provide the code to reproduce the Symile-M3 experiments from Section 5.2 of the paper (TODO link). See [here](https://github.com/rajesh-lab/symile/tree/main/src/data_processing/symile_mimic) for details on how to create the Symile-M3 dataset from scratch (TODO: maybe??).

### Dataset description

Symile-M3 is divided into three subsets, each corresponding to 2, 5, or 10 languages. Let $w$ represent the number of languages in a given subset. For each subset, an (audio, image, text) sample is generated by first drawing a short one-sentence audio clip from Common Voice (TODO link) spoken in one of $w$ languages with equal probability. From ImageNet (TODO link), an image is drawn that corresponds to one of 1,000 classes with equal probability. Finally, text containing exactly $w$ words is generated based on the drawn audio and image: one of the $w$ words in the text is the drawn image class name in the drawn audio language. The remaining $w-1$ words are randomly chosen from the ImageNet class names and written in one of the $w$ languages such that there is no overlap in language or class name across the $w$ words in the text. The words are separated by underscores, and their order is randomized. The above figure (a) shows an example of the data-generating process when $w=5$.

Notice that predicting an image from audio and text requires both inputs; relying solely on audio or text is insufficient. Thus, Symile-M3 is specifically designed to test a model's ability to capture higher-order information between the three high-dimensional data types.

### Accessing the data

As mentioned, Symile-M3 is divided into three subsets, each corresponding to 2, 5, or 10 languages. For each subset, 10M training, 500K validation, and 500K test samples were generated.

TODO: fill out instructions for accessing the data once you've added to repo.

### Saving representations

TODO

### Pre-training

The following command runs pretraining on Symile-M3:

```
(symile-env) > python main.py --experiment symile_m3 [FLAGS]
```

In addition to the [common pre-training command-line arguments](#pretrain_args), this command takes the following experiment-specific flags:

| Flag                    | Description                                         | Type   | Choices                        | Default                                                  |
|-------------------------|-----------------------------------------------------|--------|--------------------------------|----------------------------------------------------------|
| `--audio_model_id`      | Hugging Face model id for audio encoder             | str    |       |  |
| `--image_model_id`      | Hugging Face model id for image encoder             | str    | |  |
| `--text_model_id`       | Hugging Face model id for text encoder              | str    |             |  |
| `--num_langs`           | Number of languages                                 | int    | Any positive integer           | `2`                                                      |
| `--data_reference`      | Path to the data reference JSON file                | str    | Any valid file path            | `/gpfs/scratch/as16583/symile/symile/datasets/symile_m3/data_reference.json` |
| `--missingness`               | Whether to train with missingness                             | bool                | `True`, `False`                  | `False`        |
| `--missingness_prob`          | Probability with which a given modality is missing            | float  | Any float between 0.0 and 1.0  |  |

TOOD: explain missingness, and going to have to explain that the script expects the representations to be saved in advance

explain these
    --ckpt_save_dir /gpfs/scratch/as16583/ckpts/high_dim \
    --data_dir /gpfs/scratch/as16583/symile/symile/datasets/symile_m3/2W_2L \

### Evaluation

`cd` into `src/symile_data/` and set arguments in `parse_args_test()` in `args.py`. Be sure to update `--ckpt_path` and `--evaluation`.

Then run:

```
(symile-env) > python test.py
```

All checkpoints will be saved to `./ckpts/support/`.

<a name="predict_cxr"></a>
## 3. Experiment: predict CXR from ECG and labs

Note that running get_mimic_data.py for `cxr` takes about 10 minutes, for `ecg` takes about 2.5 hours, and for `labs` takes about 5 hours.

- first run get_mimic_data.py to pull data from MIMIC directories into three separate csv files: cxr_df.csv, ecg_df.csv, labs_df.csv. All of these are admissions-based df whose unique identifiers are hadm_id.
- then run create_dataset.py to create dataset.csv
- then run create_splits.py to create train.csv, etc.
- then run save_dataset_tensors.py to create dataset pt tensors in split specific directories

<a name="questions"></a>
## Questions?
For questions related to the paper, please post on alphaXiv (TODO link) for a prompt response from the authors. For questions related to the code, please open an issue in this repo or email Adriel Saporta (adriel@nyu.edu).

<a name="license"></a>
## License

TODO

<a name="citation"></a>
## Citation

TODO
