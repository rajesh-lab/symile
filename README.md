# SYMILE

<!-- **General repo structure**
- `./src/`: project library
- `./tests/`: tests for project library -->

## 0. Set up environment
<!--
#### create environment

TODO: use requirements.txt or pyproject.toml: https://github.com/pypa/packaging.python.org/issues/685#issuecomment-1321616748 and https://venthur.de/2022-12-18-python-packaging.html#:~:text=requirements.,-txt&text=txt%20are%20still%20needed%20if,same%20as%20defined%20in%20pyproject.

#### activate environment

```
> some-command
(symile-env) >
``` -->

#### install project library

From the root directory, run
```
(symile-env) > pip install -e .
```

## 1. Run binary data experiments

Run binary data experiment according to the below data generating process:

[TODO: insert screenshot, or latex for DGP from Section 5.1.]

`cd` into `src/synthetic_data/` and set experiment parameters in `args.py`. Then run:

```
(symile-env) > python main.py
```

### Mutual information and total correlation

The following command runs a script that calculates $I(\mathbf{a};\mathbf{c}), I(\mathbf{b};\mathbf{c}), I(\mathbf{a};\mathbf{b}|\mathbf{c}), I(\mathbf{c};b|\mathbf{a}), TC(\mathbf{a},\mathbf{b},\mathbf{c})$ for each $\hat{p} \in \{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0\}$.

```
(symile-env) > python informations.py [FLAGS]
```

**Required flags**
* `--d_v`: dimensionality of $\mathbf{a}$, $\mathbf{b}$, and $\mathbf{c}$.
* `--save_dir`: directory in which results will be saved. TODO: Eventually have: `Default is current directory`.

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