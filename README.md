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
(simile-env) >
``` -->

#### install project library

From the root directory, run
```
(simile-env) > pip install -e .
```

## 1. Generate data

`cd` into `src/symile_data/` and set dataset parameters in `args.py`. You'll likely want to update `--n_per_language` and `--save_path`. If you're generating data for the support classification experiment, you'll want to set `--negative_samples` to `True`.

Then run:

```
(simile-env) > python generate_data.py
```

Note that you should use this script to generate train/val/test sets separately in order to ensure that each split has the same number of samples from each template.

## 1. Run synthetic experiments

For experiment details, see [this document](https://www.overleaf.com/7416843814fymsbdxpsrxy).

`cd` into `src/synthetic_data/` and set experiment parameters in `args.py`. Then run:

```
(simile-env) > python main.py
```

### Modulo experiment

## -1. Tests

To run tests for, for example, `src/synthetic_experiments`, `cd` into `src` and run:

```
(simile-env) > pytest --pyargs synthetic_experiments
```