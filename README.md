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

## 1. Run synthetic experiments

For experiment details, see [this document](https://www.overleaf.com/7416843814fymsbdxpsrxy).

`cd` into `src/synthetic_experiments/` and set experiment parameters in `params.py`. Then run:

```
(simile-env) > python main.py
```

### Modulo experiment

## -1. Tests

To run tests for, for example, `src/synthetic_experiments`, `cd` into `src` and run:

```
(simile-env) > pytest --pyargs synthetic_experiments
```