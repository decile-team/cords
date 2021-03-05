<p align="center">
    <br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
        <img src="https://github.com/decile-team/cords/blob/2d78caf54d871976bd703f9fd14e7906264fffa1/docs/source/imgs/cords_logo.png" width="500"/>
    </br>
    <br>
        <strong> COResets and Data Subset selection </strong>
    </br>
</p>

<p align="center">
    <a href="https://github.com/decile-team/cords/blob/main/LICENSE.txt">
        <img alt="GitHub" src="https://img.shields.io/github/license/decile-team/cords?color=blue">
    </a>
    <a href="https://decile.org/">
        <img alt="Decile" src="https://img.shields.io/badge/website-online-green">
    </a>  
    <a href="https://cords.readthedocs.io/en/latest/">
        <img alt="Documentation" src="https://img.shields.io/badge/docs-passing-brightgreen">
    </a>
    <a href="#">
        <img alt="GitHub Stars" src="https://img.shields.io/github/stars/decile-team/cords">
    </a>
    <a href="#">
        <img alt="GitHub Forks" src="https://img.shields.io/github/forks/decile-team/cords">
    </a>
</p>

<h3 align="center">
<p>Reduce end to end training time from days to hours and hours to minutes using coresets and data selection.
</h3>


## In this README

- [What is CORDS?](#what-is-cords?)
- [Installation](#installation)
    - [Installing via pip](#installing-via-pip)
    - [Installing from source](#installing-from-source)
- [Getting Started](#getting-started)
- [Benchmarking Results](#benchmarking-results)
- [Publications](#publications)


## What is CORDS?

[CORDS](https://cords.readthedocs.io/en/latest/) is an efficient and scalable library for 
data efficient machine learning built on top of pytorch. 

<p align="center"><i><b>Data Efficient</b></i></p>
<p align="center"><i><b>Reduced Train Time</b></i></p>
<p align="center"><i><b>Scalable</b></i></p>

The primary purpose of CORDS is to select the right representative data subset from massive datasets. 
We use submodularity based data selection strategies to select such subsets.

CORDS implements a number of state of the art data subset selection algorithms 
and coreset algorithms. Some of the algorithms currently implemented with CORDS include:

- GLISTER [1]
- GradMatch [2]
- CRAIG [2,3]
- SubmodularSelection [4,5,6]
  - Facility Location
  - Feature Based Functions
  - Coverage
  - Diversity
- RandomSelection


## Installation

1. To install latest version of CORDS package using PyPI:

    ```python
    pip install -i https://test.pypi.org/simple/ cords
    ```

2. To install using source:

    ```bash
    git clone https://github.com/decile-team/cords.git
    cd cords
    pip install -r requirements/requirements.txt
    ```



## Getting Started

Here are some [tutorials](https://github.com/decile-team/cords/tree/main/notebooks) to get you started with CORDS. 


## Benchmarking results

The below link contains the jupyter notebook link for cifar10 timing analysis experiments

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) CIFAR10 Notebook](https://colab.research.google.com/drive/1xT6sGmDGMz8XBDmOKs5cl1cipX0Ss1sh?usp=sharing)

Results for running the benchmark on the different datasets for each strategy with different budgets for 300 epochs. The accuracy and cumulative time are calculated after 200 epochs.
The following results in the table show that training data is reduced significantly compared to training with full dataset with very less reduction in accuracy.

#### MNIST

| Strategy             | Budget (%) | Accuracy (%) | Time (hr) |
|----------------------|------------|--------------|-----------|
| Full Training        | -          | 99.356       | 0.41995   |
| GLISTER              | 10         | 99.268       | 0.10737   |
|                      | 5          | 99.28        | 0.06907   |
|                      | 3          | 99.162       | 0.3139    |
|                      | 1          | 94.71        | 0.03391   |
| GLISTER-Expore       | 10         | 99.316       | 0.07316   |
|                      | 5          | 99.262       | 0.06492   |
|                      | 3          | 99.102       | 0.0316    |
|                      | 1          | 97.604       | 0.03413   |
| GradMatch            | 10         | 99.318       | 0.07577   |
|                      | 5          | 99.282       | 0.08535   |
|                      | 3          | 99.268       | 0.04533   |
|                      | 1          | 99.026       | 0.05753   |
| GradMatch-Warm    | 10         | 99.344       | 0.07969   |
|                      | 5          | 99.272       | 0.09255   |
|                      | 3          | 99.276       | 0.04556   |
|                      | 1          | 99.134       | 0.05752   |
| GradMatchPB          | 10         | 99.276       | 0.06891   |
|                      | 5          | 99.2         | 0.06756   |
|                      | 3          | 99.984       | 0.02971   |
|                      | 1          | 98.524       | 0.03247   |
| GraddMatchPB-Warm | 10         | 99.28        | 0.06797   |
|                      | 5          | 99.214       | 0.06202   |
|                      | 3          | 99.13        | 0.03352   |
|                      | 1          | 99.754       | 0.02034   |
| Random-Online        | 10         | 99.328       | 0.07035   |
|                      | 5          | 99.294       | 0.0362    |
|                      | 3          | 99.188       | 0.01311   |
|                      | 1          | 98.92        | 0.00688   |


#### Fashion-MNIST

| Strategy             | Budget (%) | Accuracy (%) | Time (hr) |
|----------------------|------------|--------------|-----------|
| Full Training        | -          | 93.356       | 0.43896   |
| GLISTER              | 30         | 91.45        | 0.16675   |
|                      | 10         | 92.38        | 0.06565   |
|                      | 5          | 88.38        | 0.06468   |
|                      | 3          | 85.006       | 0.03129   |
|                      | 1          | 71.984       | 0.03515   |
| GLISTER-Expore       | 30         | 95.536       | 0.17465   |
|                      | 10         | 92.718       | 0.06582   |
|                      | 5          | 91.12        | 0.06395   |
|                      | 3          | 87.524       | 0.03133   |
|                      | 1          | 79.524       | 0.03511   |
| GradMatch            | 30         | 92.36        | 0.26974   |
|                      | 10         | 90.386       | 0.08464   |
|                      | 5          | 89.922       | 0.06145   |
|                      | 3          | 89.1         | 0.0534    |
|                      | 1          | 86.47        | 0.05912   |
| GradMatch-Warm    | 30         | 92.814       | 0.27953   |
|                      | 10         | 91.264       | 0.08533   |
|                      | 5          | 90.744       | 0.06173   |
|                      | 3          | 90.4484      | 0.05087   |
|                      | 1          | 87.996       | 0.05702   |
| GradMatchPB          | 30         | 93.032       | 0.30708   |
|                      | 10         | 92.506       | 0.06538   |
|                      | 5          | 91.494       | 0.03887   |
|                      | 3          | 90.596       | 0.02853   |
|                      | 1          | 88.092       | 0.0323    |
| GraddMatchPB-Warm | 30         | 93.294       | 0.29461   |
|                      | 10         | 92.714       | 0.06553   |
|                      | 5          | 92.144       | 0.03851   |
|                      | 3          | 91.352       | 0.02855   |
|                      | 1          | 89.184       | 0.02142   |
| Random-Online        | 30         | 93.298       | 0.12883   |
|                      | 10         | 93.158       | 0.04293   |
|                      | 5          | 92.836       | 0.0337    |
|                      | 3          | 92.186       | 0.01322   |
|                      | 1          | 90.52        | 0.00708   |


## Publications

[1] Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, and Rishabh Iyer, [GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning](https://arxiv.org/abs/2012.10630), 35th AAAI Conference on Artificial Intelligence, AAAI 2021

[2] S Durga, Krishnateja Killamsetty, Abir De, Ganesh Ramakrishnan, Baharan Mirzasoleiman, Rishabh Iyer, Grad-Match: A Gradient Matching based Data Selection Framework for Efficient Learning

[3] Baharan Mirzasoleiman, Jeff Bilmes, and Jure Leskovec. [Coresets for Data-efficient Training of Machine Learning Models](https://arxiv.org/abs/1906.01827). In International Conference on Machine Learning (ICML), July 2020

[4] Kai Wei, Rishabh Iyer, Jeff Bilmes, [Submodularity in Data Subset Selection and Active Learning](http://proceedings.mlr.press/v37/wei15-supp.pdf), International Conference on Machine Learning (ICML) 2015

[5] Vishal Kaushal, Rishabh Iyer, Suraj Kothiwade, Rohan Mahadev, Khoshrav Doctor, and Ganesh Ramakrishnan, [Learning From Less Data: A Unified Data Subset Selection and Active Learning Framework for Computer Vision](https://arxiv.org/abs/1901.01151), 7th IEEE Winter Conference on Applications of Computer Vision (WACV), 2019 Hawaii, USA

[6] Wei, Kai, et al. [Submodular subset selection for large-scale speech training data](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.496.6287&rep=rep1&type=pdf), 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2014.
