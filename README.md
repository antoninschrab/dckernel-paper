# Reproducibility code for dcMMD & dcHSIC 

# Robust Kernel Hypothesis Testing under Data Corruption

This GitHub repository contains the code for the reproducible experiments presented in our paper 
[Robust Kernel Hypothesis Testing under Data Corruption](https://arxiv.org/abs/2405.19912).

The code is written in [JAX](https://jax.readthedocs.io/) which can leverage the architecture of GPUs to provide considerable computational speedups.

## Installation

In a chosen directory, clone the repository and change to its directory by executing 
```bash
git clone git@github.com:antoninschrab/dckernel-paper.git
cd dckernel-paper
```
We then recommend creating a `conda` environment with the required dependencies:
  ```bash
  conda create -n dckernel-env
  conda activate dckernel-env
  # install JAX for GPU:
  pip install -U "jax[cuda12]"
  # or install JAX for CPU:
  pip install -U "jax[cpu]"
  # install all other dependencies
  conda install numpy scipy scikit-learn matplotlib tqdm
  ```

To run only dcMMD and dcHSIC it is sufficient to only install JAX as explained in our [dckernel](https://github.com/antoninschrab/dckernel) repository. 

## Reproducibility of the experiments

The code to reproduce the experiments of the paper can be found in the [experiments.ipynb](experiments.ipynb) notebook.

For the experiments, the results and figures are saved in the [results](results) and [figures](figures) directories, respectively. 

## How to use dcMMD and dcHSIC in practice?

Our proposed dcMMD and dcHSIC tests are implemented in [dctests.py](dctests.py).

To use our tests in practice, we recommend using our `dckernel` package which is available on the [dckernel](https://github.com/antoninschrab/dckernel) repository. 
It can be installed by running
```bash
pip install git+https://github.com/antoninschrab/dckernel.git
```
Installation instructions and example code are available on the [dckernel](https://github.com/antoninschrab/dckernel) repository. 

We also illustrate how to use the tests in the demo section of the notebook [experiments.ipynb](experiments.ipynb).

## References

- DP tests: [repository](https://github.com/antoninschrab/dpkernel/), [paper](https://arxiv.org/abs/2310.19043)
- IMDb dataset: [repository](https://ai.stanford.edu/~amaas/data/sentiment/), [paper](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)

## Contact

If you have any issues running our code, please do not hesitate to contact [Antonin Schrab](https://antoninschrab.github.io).

## Affiliations

Centre for Artificial Intelligence, Department of Computer Science, University College London

Gatsby Computational Neuroscience Unit, University College London

Inria London

## Bibtex

```
@unpublished{schrab2024robust,
title={Robust Kernel Hypothesis Testing under Data Corruption}, 
author={Antonin Schrab and Ilmun Kim},
year={2024},
url = {https://arxiv.org/abs/2405.19912},
eprint={2405.19912},
archivePrefix={arXiv},
primaryClass={stat.ML}
}
```

## License

MIT License (see [LICENSE](LICENSE)).

## Related tests

- [mmdagg](https://github.com/antoninschrab/mmdagg/): MMD Aggregated MMDAgg test
- [ksdagg](https://github.com/antoninschrab/ksdagg/): KSD Aggregated KSDAgg test
- [agginc](https://github.com/antoninschrab/agginc/): Efficient MMDAggInc HSICAggInc KSDAggInc tests
- [mmdfuse](https://github.com/antoninschrab/mmdfuse/): MMD-Fuse test
- [dpkernel](https://github.com/antoninschrab/dpkernel/): Differentially private dpMMD dpHSIC tests
