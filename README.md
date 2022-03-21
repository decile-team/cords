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
- [Installation](#installation)
- [Next Steps](#next-steps)
- [Tutorials](#tutorials)
- [Documentation](#documentation)



## What is CORDS?

[CORDS](https://cords.readthedocs.io/en/latest/) is COReset and Data Selection library for making machine learning time, energy, cost, and compute efficient. CORDS is built on top of pytorch. Deep Learning systems are extremely compute intensive today with large turn around times, energy inefficiencies, higher costs and resourse requirements [1,2]. CORDS is an effort to make deep learning more energy, cost, resource and time efficient while not sacrificing accuracy. The following are the goals CORDS tries to achieve:

<p align="center"><i><b>Data Efficiency</b></i></p>
<p align="center"><i><b>Reducing End to End Training Time</b></i></p>
<p align="center"><i><b>Reducing Energy Requirement</b></i></p>
<p align="center"><i><b>Faster Hyper-parameter tuning </b></i></p>
<p align="center"><i><b>Reducing Resource (GPU) Requirement and Costs</b></i></p>


The primary purpose of CORDS is to select the right representative data subsets from massive datasets, and it does so iteratively. CORDS uses some recent advances in data subset selection and particularly, ideas of coresets and submodularity select such subsets. CORDS implements a number of state of the art data subset selection algorithms 
and coreset algorithms. Some of the algorithms currently implemented with CORDS include:

- [GLISTER [3]](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.supervisedlearning.html#module-cords.selectionstrategies.supervisedlearning.glisterstrategy)
- [GradMatch [4]](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.supervisedlearning.html#module-cords.selectionstrategies.supervisedlearning.ompgradmatchstrategy)
- [CRAIG [4,5]](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.supervisedlearning.html#module-cords.selectionstrategies.supervisedlearning.craigstrategy)
- [SubmodularSelection [6,7,8]](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.supervisedlearning.html#module-cords.selectionstrategies.supervisedlearning.submodularselectionstrategy) (Facility Location, Feature Based Functions, Coverage, Diversity)
- [RandomSelection](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.supervisedlearning.html#module-cords.selectionstrategies.supervisedlearning.randomstrategy)

We are continuously incorporating newer and better algorithms into CORDS. Some of the features of CORDS includes:

- Reproducability of SOTA in Data Selection and Coresets: Enable easy reproducability of SOTA described above. We are trying to also add more algorithms so if you have an algorithm you would like us to include, please let us know,.
- Benchmarking: We have benchmarked CORDS (and the algorithms present right now) on several datasets including CIFAR-10, CIFAR-100, MNIST, SVHN and ImageNet. 
- Ease of Use: One of the main goals of CORDS is that it is easy to use and add to CORDS. Feel free to contribute to CORDS!
- Modular design: The data selection algorithms are separate from the training loop, thereby enabling modular design and also varied scenarios of utility.
- Broad number of usecases: CORDS is currently implemented for simple image classification tasks and hyperparameter tuning, but we are working on integrating a number of additional use cases like object detection, speech recognition, semi-supervised learning, Auto-ML, etc.

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


## Next Steps

## Tutorials
We have added example python code and tutorial notebooks under examples folder.
## Documentation

The documentation for the latest version of CORDS can always be found [here](https://cords.readthedocs.io/en/latest/).


