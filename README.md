# Manifold lifting: scaling MCMC to the vanishing noise regime

Code accompanying the paper [*Manifold lifting: scaling MCMC to the vanishing noise regime*](https://arxiv.org/abs/2003.03950).

**Abstract**: Standard Markov chain Monte Carlo methods struggle to explore distributions that are concentrated in the neighbourhood of low-dimensional structures. These pathologies naturally occur in a number of situations. For example, they are common to Bayesian inverse problem modelling and Bayesian neural networks, when observational data are highly informative, or when a subset of the statistical parameters of interest are non-identifiable. In this paper, we propose a strategy that transforms the original sampling problem into the task of exploring a distribution supported on a manifold embedded in a higher dimensional space; in contrast to the original posterior this lifted distribution remains diffuse in the vanishing noise limit. We employ a constrained Hamiltonian Monte Carlo method which exploits the manifold geometry of this lifted distribution, to perform efficient approximate inference. We demonstrate in several numerical experiments that, contrarily to competing approaches, the sampling efficiency of our proposed methodology does not degenerate as the target distribution to be explored concentrates near low dimensional structures.

## Notebook

For a complete example of applying the method described in the paper to perform inference in a two-dimensional example and accompanying explanatory notes see the Jupyter notebook linked below. The manifold MCMC methods in the Python package [*Mici*](https://github.com/matt-graham/mici) are used for inference.

<table>
  <tr>
    <th colspan="2"><img src='https://raw.githubusercontent.com/jupyter/design/master/logos/Favicon/favicon.svg?sanitize=true' width="15" style="vertical-align:text-bottom; margin-right: 5px;"/> <a href="notebooks/Two-dimensional_example.ipynb">Two-dimensional_example.ipynb</a></th>
  </tr>
  <tr>
    <td>Open non-interactive version with nbviewer</td>
    <td>
      <a href="https://nbviewer.jupyter.org/github/thiery-lab/manifold_lifting/blob/master/notebooks/Two-dimensional_example.ipynb">
        <img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg?sanitize=true" width="109" alt="Render with nbviewer"  style="vertical-align:text-bottom" />
      </a>
    </td>
  </tr>
  <tr>
    <td>Open interactive version with Binder</td>
    <td>
      <a href="https://mybinder.org/v2/gh/thiery-lab/manifold_lifting/master?filepath=notebooks%2FTwo-dimensional_example.ipynb">
        <img src="https://mybinder.org/badge_logo.svg" alt="Launch with Binder"  style="vertical-align:text-bottom"/>
      </a>
    </td>
  </tr>
  <tr>
    <td>Open interactive version with Google Colab</td>
    <td>
      <a href="https://colab.research.google.com/github/thiery-lab/manifold_lifting/blob/master/notebooks/Two-dimensional_example.ipynb">
        <img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align:text-bottom">
       </a> 
    </td>
  </tr>
</table>

## Local installation

To install the dependencies to run the notebook locally, first create a local clone of the repository

```bash
git clone https://github.com/thiery-lab/manifold_lifting.git
```

Then either create a new Python 3.6+ environment using your environment manager of choice (e.g. [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands), [`virtualenv`](https://virtualenv.pypa.io/en/latest/userguide/#usage), [`venv`](https://docs.python.org/3/library/venv.html#creating-virtual-environments), [`pipenv`](https://pipenv.kennethreitz.org/en/latest/install/#installing-packages-for-your-project)) or activate the existing environment you wish to use.

To install the dependencies, from within the `manifold_lifting` directory run

```bash
pip install -r requirements.txt
```

## Citation

To cite the pre-print the following `bibtex` entry can be used

```bibtex
@misc{au2020manifold,
    title={Manifold lifting: scaling MCMC to the vanishing noise regime},
    author={Khai Xiang Au and Matthew M. Graham and Alexandre H. Thiery},
    year={2020},
    eprint={2003.03950},
    archivePrefix={arXiv},
    primaryClass={stat.CO}
}
```
