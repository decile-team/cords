<p align="center">
    <br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
        <img src="https://github.com/decile-team/cords/blob/main/docs/source/imgs/cords_logo.png" width="500"/>
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
<p>Reduce end to end training time from days to hours (or hours to minutes), and energy requirements/costs by an order of magnitude using coresets and data selection.
</h3>


## In this README

- [In this README](#in-this-readme)
- [What is CORDS?](#what-is-cords)
  - [SpeedUps in Supervised Learning](#speedups-in-supervised-learning)
  - [SpeedUps in Semi-supervised Learning](#speedups-in-semi-supervised-learning)
  - [SpeedUps in Hyperparameter Tuning](#speedups-in-hyperparameter-tuning)
- [Highlights](#highlights)
- [Installation](#installation)
- [Next Steps](#next-steps)
- [Tutorials](#tutorials)
- [Documentation](#documentation)
- [Publications](#publications)

## What is CORDS?

[CORDS](https://cords.readthedocs.io/en/latest/) is COReset and Data Selection library for making machine learning time, energy, cost, and compute efficient. CORDS is built on top of PyTorch. Today, deep learning systems are extremely compute-intensive, with significant turnaround times, energy inefficiencies, higher costs, and resource requirements [1,2]. CORDS is an effort to make deep learning more energy, cost, resource, and time-efficient while not sacrificing accuracy. The following are the goals CORDS tries to achieve:

<p align="center"><i><b>Data Efficiency</b></i></p>
<p align="center"><i><b>Reducing End to End Training Time</b></i></p>
<p align="center"><i><b>Reducing Energy Requirement</b></i></p>
<p align="center"><i><b>Faster Hyper-parameter tuning </b></i></p>
<p align="center"><i><b>Reducing Resource (GPU) Requirement and Costs</b></i></p>

The primary purpose of CORDS is to select the suitable representative data subsets from massive datasets, and it does so iteratively. CORDS uses recent advances in data subset selection, particularly ideas of coresets and submodularity select such subsets. CORDS implements several state-of-the-art data subset/coreset selection algorithms for efficient supervised learning(SL) and semi-supervised learning(SSL). Some of the algorithms currently implemented with CORDS include:

For Supervised Learning:
- [GLISTER](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.SL.html#module-cords.selectionstrategies.SL.glisterstrategy)
- [GradMatch](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.SL.html#module-cords.selectionstrategies.SL.gradmatchstrategy)
- [CRAIG](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.SL.html#module-cords.selectionstrategies.SL.craigstrategy)
- [SubmodularSelection](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.SL.html#module-cords.selectionstrategies.SL.submodularselectionstrategy) (Facility Location, Feature Based Functions, Coverage, Diversity)
- [RandomSelection](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.SL.html#module-cords.selectionstrategies.SL.randomstrategy)

For Semi-supervised Learning:
- [RETRIEVE](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.SSL.html#module-cords.selectionstrategies.SSL.retrievestrategy)
- [GradMatch](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.SSL.html#module-cords.selectionstrategies.SSL.gradmatchstrategy)
- [CRAIG](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.SSL.html#module-cords.selectionstrategies.SSL.craigstrategy)
- [RandomSelection](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.SSL.html#module-cords.selectionstrategies.SL.randomstrategy)

We are continuously incorporating newer and better algorithms into CORDS. Some of the features of CORDS includes:

- Reproducibility of SOTA in Data Selection and Coresets: Enable easy reproducibility of SOTA described above. We are trying also to add more algorithms, so if you have an algorithm you would like us to include, please let us know,
- Benchmarking: We have benchmarked CORDS (and the algorithms present right now) on several datasets, including CIFAR-10, CIFAR-100, MNIST, SVHN, and ImageNet. 
- Ease of Use: One of the main goals of CORDS is that it is easy to use and add to CORDS. Feel free to contribute to CORDS!
- Modular design: The data selection algorithms are directly incorporated into data loaders, allowing one to use their own training loop for varied utility scenarios. 
- A broad number of use cases: CORDS is currently implemented for simple image classification tasks and hyperparameter tuning, but we are working on integrating several additional use cases like  Auto-ML, object detection, speech recognition, semi-supervised learning, etc.

To achieve significantly faster speedups, one can replace the subset selection data loaders while keeping the training algorithm the same. Look at the speedups one can achieve below:

### SpeedUps in Supervised Learning

<p align="center">
    <br>
        <img src="https://github.com/decile-team/cords/blob/main/docs/source/imgs/sl_speedups.jpg" height="400" width="700"/>
    </br>
</p>

### SpeedUps in Semi-supervised Learning

<p align="center">
    <br>
        <img src="https://github.com/decile-team/cords/blob/main/docs/source/imgs/ssl_speedups.png" height="400" width="700"/>
    </br>
</p>

### SpeedUps in Hyperparameter Tuning

<p align="center">
    <br>
        <img src="https://github.com/decile-team/cords/blob/main/docs/source/imgs/hpo_speedups.png" height="400" width="700"/>
    </br>
</p>

## Highlights
- 3x to 5x speedups, cost reduction, and energy reductions in the training of deep models in supervised learning
- 3x+ speedups, cost/energy reduction for deep model training in semi-supervised learning
- 3x to 30x speedups and cost/energy reduction for Hyper-parameter tuning using subset selection with SOTA schedulers (Hyperband and ASHA) and algorithms (TPE, Random)

## Installation

1. To install the latest version of the CORDS package using PyPI:

    ```python
    pip install -i https://test.pypi.org/simple/ cords
    ```

2. To install using the source:

    ```bash
    git clone https://github.com/decile-team/cords.git
    cd cords
    pip install -r requirements/requirements.txt
    ```

## Next Steps

## Tutorials
We have added example python code and tutorial notebooks under the examples folder. See [this link](https://github.com/decile-team/cords/tree/main/examples)

## Documentation

The documentation for the latest version of CORDS can always be found [here](https://cords.readthedocs.io/en/latest/).

## Publications

[1] Schwartz, Roy, et al. "Green ai." arXiv preprint arXiv:1907.10597 (2019).

[2] Strubell, Emma, Ananya Ganesh, and Andrew McCallum. "Energy and policy considerations for deep learning in NLP." In ACL 2019.

[3] Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, and Rishabh Iyer, [GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning](https://arxiv.org/abs/2012.10630), 35th AAAI Conference on Artificial Intelligence, AAAI 2021

[4] Krishnateja Killamsetty, Durga Sivasubramanian, Abir De, Ganesh Ramakrishnan, Baharan Mirzasoleiman, Rishabh Iyer, [Grad-Match: A Gradient Matching based Data Selection Framework for Efficient Learning](https://arxiv.org/abs/2103.00123), 2021

[5] Baharan Mirzasoleiman, Jeff Bilmes, and Jure Leskovec. [Coresets for Data-efficient Training of Machine Learning Models](https://arxiv.org/abs/1906.01827). In International Conference on Machine Learning (ICML), July 2020

[6] Kai Wei, Rishabh Iyer, Jeff Bilmes, [Submodularity in Data Subset Selection and Active Learning](http://proceedings.mlr.press/v37/wei15-supp.pdf), International Conference on Machine Learning (ICML) 2015

[7] Vishal Kaushal, Rishabh Iyer, Suraj Kothiwade, Rohan Mahadev, Khoshrav Doctor, and Ganesh Ramakrishnan, [Learning From Less Data: A Unified Data Subset Selection and Active Learning Framework for Computer Vision](https://arxiv.org/abs/1901.01151), 7th IEEE Winter Conference on Applications of Computer Vision (WACV), 2019 Hawaii, USA

[8] Wei, Kai, et al. [Submodular subset selection for large-scale speech training data](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.496.6287&rep=rep1&type=pdf), 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2014.

