# Manifold lifting: scaling Markov chain Monte Carlo to the vanishing noise regime

[![DOI](https://zenodo.org/badge/246543695.svg)](https://zenodo.org/badge/latestdoi/246543695)
[![arXiv:1807.02356](https://img.shields.io/badge/arXiv-2003.03950-B31B1B.svg)](https://arxiv.org/abs/2003.03950)
[![DOI:10.1093/jrsssb/qkad023](https://img.shields.io/badge/Journal_article-10.1093/jrsssb/qkad023-d4a519.svg)](https://doi.org/10.1093/jrsssb/qkad023)

Code accompanying the paper *Manifold lifting: scaling Markov chain Monte Carlo to the vanishing noise regime*.

**Abstract**: Standard Markov chain Monte Carlo methods struggle to explore distributions that concentrate in the neighbourhood of low-dimensional submanifolds. This pathology naturally occurs in Bayesian inference settings when there is a high signal-to-noise ratio in the observational data but the model is inherently over-parametrised or non-identifiable. In this paper, we propose a strategy that transforms the original sampling problem into the task of exploring a distribution supported on a manifold embedded in a higher-dimensional space; in contrast to the original posterior this lifted distribution remains diffuse in the limit of vanishing observation noise. We employ a constrained Hamiltonian Monte Carlo method, which exploits the geometry of this lifted distribution, to perform efficient approximate inference. We demonstrate in numerical experiments that, contrarily to competing approaches, the sampling efficiency of our proposed methodology does not degenerate as the target distribution to be explored concentrates near low-dimensional submanifolds.

## Installation

The `mlift` package requires Python 3.8 or above. To install the `mlift` package and its dependencies into an existing Python environment run

```bash
pip install git+https://github.com/thiery-lab/manifold_lifting.git
```

The `mlift.pde` module and associated example model in `mlift.example_models.poisson` additionally require [FEniCS (v2019.10)](https://fenicsproject.org/download/archive/) and [`scikit-sparse`](https://scikit-sparse.readthedocs.io/en/latest/overview.html#installation) to be installed. Note that the [`fenics` metapackage on PyPI](https://pypi.org/project/fenics/) does not install the required binary dependencies and so FEniCS should instead be separately installed by [following one of the methods listed in the project's installation instructions](https://fenicsproject.org/download/archive/).

Alternatively a `conda` environment `man-lift` containing the required dependencies to run all of the experiments (including the `mlift` package) can be created from the provided [`environment.yml`](environment.yml) file by running

```bash
conda env create -f environment.yml
```

## Experiment scripts

A number of scripts for reproducing the numerical experiments used to produce the figures in the paper are provided in the `scripts` directory. To run these scripts the `mlift` package and its dependencies need to be installed in a local Python 3.9+ environment [as described above](#installation). The default settings of the scripts assume they are run from the _top-level_ directory of a clone of the repository. Pass a `--help` argument to any of the scripts to see a description of what the script does and the available environment variables for configuring the behaviour of the script.

## Example notebook

For a complete example of applying the method described in the paper to perform inference in a two-dimensional example and accompanying explanatory notes see the Jupyter notebook linked below. The manifold MCMC methods in the Python package [*Mici*](https://github.com/matt-graham/mici) are used for inference.

<table>
  <tr>
    <th colspan="2"><img src='https://raw.githubusercontent.com/jupyter/design/main/logos/Favicon/favicon.svg?sanitize=true' width="15" style="vertical-align:text-bottom; margin-right: 5px;"/> <a href="notebooks/Two-dimensional_example.ipynb"> Two-dimensional_example.ipynb</a></th>
  </tr>
  <tr>
    <td>Open non-interactive version with nbviewer</td>
    <td>
      <a href="https://nbviewer.jupyter.org/github/thiery-lab/manifold_lifting/blob/main/notebooks/Two-dimensional_example.ipynb">
        <img src="https://raw.githubusercontent.com/jupyter/design/main/logos/Badges/nbviewer_badge.svg?sanitize=true" width="109" alt="Render with nbviewer"  style="vertical-align:text-bottom" />
      </a>
    </td>
  </tr>
  <tr>
    <td>Open interactive version with Binder</td>
    <td>
      <a href="https://mybinder.org/v2/gh/thiery-lab/manifold_lifting/main?filepath=notebooks%2FTwo-dimensional_example.ipynb">
        <img src="https://mybinder.org/badge_logo.svg" alt="Launch with Binder"  style="vertical-align:text-bottom"/>
      </a>
    </td>
  </tr>
  <tr>
    <td>Open interactive version with Google Colab</td>
    <td>
      <a href="https://colab.research.google.com/github/thiery-lab/manifold_lifting/blob/main/notebooks/Two-dimensional_example.ipynb">
        <img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align:text-bottom">
       </a>
    </td>
  </tr>
</table>

## Citation

To cite the paper the following `bibtex` entry can be used

```bibtex
@article{au2023manifold,
    author = {Au, Khai Xiang and Graham, Matthew M and Thiery, Alexandre H},
    title = "{Manifold lifting: scaling Markov chain Monte Carlo to the vanishing noise regime}",
    journal = {Journal of the Royal Statistical Society Series B: Statistical Methodology},
    volume = {85},
    number = {3},
    pages = {757-782},
    year = {2023},
    month = {04},
    issn = {1369-7412},
    doi = {10.1093/jrsssb/qkad023},
    url = {https://doi.org/10.1093/jrsssb/qkad023},
    eprint = {https://academic.oup.com/jrsssb/article-pdf/85/3/757/50859790/qkad023.pdf},
}
```
