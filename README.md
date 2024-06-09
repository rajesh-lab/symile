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
- [Command-line arguments](#common_args)
- [Binary XOR experiments](#binary_xor)
- [Symile-M3 experiments](#symile_m3)
- [Symile-MIMIC experiments](#cxr_prediction)
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

<a name="common_args"></a>
## Command-line arguments

The following command-line arguments are common to all three sets of experiments (Binary XOR, Symile-M3, and Symile-MIMIC) and can be specified when running `main.py`.

TODO: make sure wandb set to false by default, and set symile to default loss function, and make sure use_seed is False by default. also update bash script examples so people have default... or include a txt with examples. maybe make save ckpts the default be current directory

| Flag                          | Description                                                   | Type                | Choices                          | Default |
|-------------------------------|---------------------------------------------------------------|---------------------|----------------------------------|---------------|
| `--experiment`                | Experiment identifier                                         | str                 | `binary_xor`, `symile_m3`, `symile_mimic` |               |
| `--batch_sz_train`            | Batch size for training                                       | int                 |                                  |               |
| `--batch_sz_val`              | Batch size for validation                                     | int                 |                                  |               |
| `--batch_sz_test`             | Batch size for testing                                        | int                 |                                  |               |
| `--epochs`                    | Number of training epochs                                     | int                 |                                  |               |
| `--check_val_every_n_epoch`   | Frequency of validation checks (in epochs)                    | int                 |                                  |               |
| `--drop_last`                 | Drop the last incomplete training batch if the training set is not divisible by batch size | bool | `True`, `False`                  |               |
| `--lr`                        | Learning rate                                                 | float               |                                  |               |
| `--logit_scale_init`          | Initial value for the logit scale, which is the temperature parameter $\tau$ is directly optimized during training as a multiplicative scalar to avoid having to tune it as a hyperparameter | float | | |
| `--negative_sampling`         | Negative sampling strategy: $O(N)$ or $O(N^2)$                | str                 | `n`, `n_squared`                 |               |
| `--loss_fn`                   | Loss function to use                                          | str                 | `symile`, `clip`        |  `symile`             |
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
(symile-env) > python main.py [FLAGS]
```

In addition to the [common command-line arguments](#common_args) outlined above, this command takes the following experiment-specific flags:

| Flag        | Description                               | Type   | Choices           | Default |
|-------------|-------------------------------------------|--------|-------------------|---------|
| `--train_n` | Number of training samples to draw        | int    |  |    |
| `--val_n`   | Number of validation samples to draw      | int    |  |     |
| `--test_n`  | Number of test samples to draw            | int    |  |     |
| `--d_v`     | Dimensionality of the input vectors $\mathbf{a}$, $\mathbf{b}$, and $\mathbf{c}$            | int    |  |       |
| `--d_r`     | Dimensionality of the learned representation vectors   | int    |  |       |

### Calculate information terms

We also include the code to track the changing information dynamics between the variables $\mathbf{a}$, $\mathbf{b}$, and $\mathbf{c}$ as $\hat{p}$ moves from 0 to 1 (Figure 3b in the paper (TODO link)).
Specifically, the following command calculates $\mathbf{I}(\mathbf{a};\mathbf{c})$, $\mathbf{I}(\mathbf{b};\mathbf{c})$, $`\mathbf{I}(\mathbf{a};\mathbf{b}\,|\,\mathbf{c})`$, $`\mathbf{I}(\mathbf{c};\mathbf{b}\,|\,\mathbf{a})`$, and $\mathbf{TC}(\mathbf{a},\mathbf{b},\mathbf{c})$ for each $\hat{p}$ in $`\{0.0, 0.1,0.2,\dots,1.0\}`$:

```
(symile-env) > python informations.py --d_v <input_vector_dim> --save_dir <path/to/save_dir>
```

Note that running this script for `d_v = 5` takes about 1.5 hours.

## 2. Run SYMILE dataset experiments

### Generate data

#### Google Cloud APIs

We use the Google Cloud Translation and Text-to-Speech APIs to create our dataset. You'll need the following Google client libraries to run `generate_data.py`.

##### Translate

https://cloud.google.com/translate/docs/setup

just need basic client libraries

##### Text-to-Speech

Follow the instruction here (although can we just package it up in the environment?): https://cloud.google.com/text-to-speech/docs/libraries (probably will need to still install and initialize gcloud CLI and create credential file like in here: https://cloud.google.com/docs/authentication/provide-credentials-adc)

- Include instructions for how to install google translate/tts

#### Create splits from ImageNet

#### Create datasets

`cd` into `src/symile_data/` and set dataset parameters in `args.py`. You'll likely want to update `--n_per_language` and `--save_path`. If you're generating data for the support classification experiment, you'll want to set `--negative_samples` to `True`.

Then run:

```
(symile-env) > python generate_data.py
```

Note that you should use this script to generate train/val/test sets separately in order to ensure that each split has the same number of samples from each template.

### Pre-train

`cd` into `src/symile_data/` and set arguments in `parse_args_pretrain()` in `args.py`.

Then run:

```
(symile-env) > python pretrain.py
```

All checkpoints will be saved to `./ckpts/pretrain/`.

### Evaluation: zero-shot classification

`cd` into `src/symile_data/` and set arguments in `parse_args_test()` in `args.py`. Be sure to update `--ckpt_path` and `--evaluation`.

Then run:

```
(symile-env) > python test.py
```

### Evaluation: in-support classification

`cd` into `src/symile_data/` and set arguments in `parse_args_test()` in `args.py`. Be sure to update `--ckpt_path` and `--evaluation`.

Then run:

```
(symile-env) > python test.py
```

All checkpoints will be saved to `./ckpts/support/`.

Note that for support classification model fitting and testing are both in this scipt: because we run trainer.test() directly after trainer.fit(), trainer.test() automatically loads the best weights from training. As a result, DDP should not be used when running this script (i.e. no more than a single GPU should be used). During trainer.test(), it is recommended to use Trainer(devices=1) to ensure each sample/batch gets evaluated exactly once. Otherwise, multi-device settings use `DistributedSampler` that replicates some samples to make sure all devices have same batch size in case of uneven inputs.

<a name="predict_cxr"></a>
## 3. Experiment: predict CXR from ECG and labs

Note that running get_mimic_data.py for `cxr` takes about 10 minutes, for `ecg` takes about 2.5 hours, and for `labs` takes about 5 hours.

- first run get_mimic_data.py to pull data from MIMIC directories into three separate csv files: cxr_df.csv, ecg_df.csv, labs_df.csv. All of these are admissions-based df whose unique identifiers are hadm_id.
- then run create_dataset.py to create dataset.csv
- then run create_splits.py to create train.csv, etc.
- then run save_dataset_tensors.py to create dataset pt tensors in split specific directories

<a name="license"></a>
## License

TODO

<a name="citation"></a>
## Citation

TODO
