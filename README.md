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
- [Highlights](#highlights)
- [Starting with CORDS](#starting-with-cords)
  - [Pip Installation](#pip-installation)
  - [From Git Repository](#from-git-repository)
  - [First Steps](#first-steps)
  - [Using subset selection based data loaders](#using-subset-selection-based-data-loaders)
- [Applications](#applications)
  - [Efficient Hyper-parameter Optimization(HPO)](#efficient-hyper-parameter-optimizationhpo)
- [Speedups achieved using CORDS](#speedups-achieved-using-cords)
  - [SpeedUps in Supervised Learning](#speedups-in-supervised-learning)
  - [SpeedUps in Semi-supervised Learning](#speedups-in-semi-supervised-learning)
  - [SpeedUps in Hyperparameter Tuning](#speedups-in-hyperparameter-tuning)
- [Tutorials](#tutorials)
- [Documentation](#documentation)
- [Mailing List](#mailing-list)
- [Acknowledgment](#acknowledgment)
- [Team](#team)
- [Resources](#resources)
- [Publications](#publications)

## What is CORDS?

[CORDS](https://cords.readthedocs.io/en/latest/) is COReset and Data Selection library for making machine learning time, energy, cost, and compute efficient. CORDS is built on top of PyTorch. Today, deep learning systems are extremely compute-intensive, with significant turnaround times, energy inefficiencies, higher costs, and resource requirements [1,2]. CORDS is an effort to make deep learning more energy, cost, resource, and time-efficient while not sacrificing accuracy. The following are the goals CORDS tries to achieve:

<p align="center"><i><b>Data Efficiency</b></i></p>
<p align="center"><i><b>Reducing End to End Training Time</b></i></p>
<p align="center"><i><b>Reducing Energy Requirement</b></i></p>
<p align="center"><i><b>Faster Hyper-parameter tuning </b></i></p>
<p align="center"><i><b>Reducing Resource (GPU) Requirement and Costs</b></i></p>

The primary purpose of CORDS is to select the suitable representative data subsets from massive datasets, and it does so iteratively. CORDS uses recent advances in data subset selection, particularly ideas of coresets and submodularity select such subsets. CORDS implements several state-of-the-art data subset/coreset selection algorithms for efficient supervised learning(SL) and semi-supervised learning(SSL). 

Some of the algorithms currently implemented with CORDS include:

For Efficient and Robust Supervised Learning:
- [GLISTER](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.SL.html#module-cords.selectionstrategies.SL.glisterstrategy)
- [GradMatch](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.SL.html#module-cords.selectionstrategies.SL.gradmatchstrategy)
- [CRAIG](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.SL.html#module-cords.selectionstrategies.SL.craigstrategy)
- [SubmodularSelection](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.SL.html#module-cords.selectionstrategies.SL.submodularselectionstrategy) (Facility Location, Feature Based Functions, Coverage, Diversity)
- [RandomSelection](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.SL.html#module-cords.selectionstrategies.SL.randomstrategy)

For Efficient and Robust Semi-supervised Learning:
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

## Highlights
- 3x to 5x speedups, cost reduction, and energy reductions in the training of deep models in supervised learning
- 3x+ speedups, cost/energy reduction for deep model training in semi-supervised learning
- 3x to 30x speedups and cost/energy reduction for Hyper-parameter tuning using subset selection with SOTA schedulers (Hyperband and ASHA) and algorithms (TPE, Random)

## Starting with CORDS

### Pip Installation

To install the latest version of the CORDS package using PyPI:

```python
pip install -i https://test.pypi.org/simple/cords
```

### From Git Repository
To install using the source:

```bash
git clone https://github.com/decile-team/cords.git
cd cords
pip install -r requirements/requirements.txt
```

### First Steps

To better understand CORDS's functionality, we have provided example Jupyter notebooks and python code in the [examples](https://github.com/decile-team/cords/tree/main/examples) folder, which can be easily executed by using Google Colab. We also provide a simple SL, SSL, and HPO training loops that runs experiments using a provided configuration file. To run this loop, you can look into following code examples:

Using default supervised training loop,
```python
from train_sl import TrainClassifier
from cords.utils.config_utils import load_config_data

config_file = '/content/cords/configs/SL/config_glister_cifar10.py'
cfg = load_config_data(config_file)
clf = TrainClassifier(cfg)
clf.train()
```

Using default semi-supervised training loop,
```python
from train_ssl import TrainClassifier
from cords.utils.config_utils import load_config_data

config_file = '/content/cords/configs/SSL/config_retrieve-warm_vat_cifar10.py'
cfg = load_config_data(config_file)
clf = TrainClassifier(cfg)
clf.train()
```

You can use the default configurations that we have provided in the configs folder, or you can make a custom configuration. For making your custom configuration file for training, please refer to [CORDS Configuration File Documentation](https://cords.readthedocs.io/en/latest/strategies/cords.configs.html).

### Using subset selection based data loaders
Create a subset selection based data loader at train time and use the subset selection based data loader with your own training loop.

Essentially, with subset selection-based data loaders, it is pretty straightforward to use subset selection strategies directly 
because they are integrated directly into subset data loaders; this allows users to use subset selection strategies directly by 
using their respective subset selection data loaders.

Below is an example that shows the subset selection process is simplified by just calling a data loader in supervised learning setting,

```python
    dss_args = dict(model=model,
                    loss=criterion_nored,
                    eta=0.01,
                    num_classes=10,
                    num_epochs=300,
                    device='cuda',
                    fraction=0.1,
                    select_every=20,
                    kappa=0,
                    linear_layer=False,
                    selection_type='SL',
                    greedy='Stochastic')
    dss_args = DotMap(dss_args)

    dataloader = GLISTERDataLoader(trainloader, valloader, dss_args, logger, 
                                    batch_size=20, 
                                    shuffle=True,
                                    pin_memory=False)
    
    for epoch in range(num_epochs):
        for _, (inputs, targets, weights) in enumerate(dataloader):
            """
            Standard PyTorch training loop using weighted loss
            
            Our training loop differs from the standard PyTorch training loop in that along with 
            data samples and their associated target labels; we also have additional sample weight
            information from the subset data loader, which can be used to calculate the weighted 
            loss for gradient descent. We can calculate the weighted loss by using default PyTorch
            loss functions with no reduction.
            """
```

In our current version, we deployed subset selection data loaders in supervised learning and semi-supervised learning settings.


## Applications

### Efficient Hyper-parameter Optimization(HPO)
The subset selection strategies for efficient supervised learning in CORDS allow one to train models faster. We can use the faster model training using data subsets for quicker configuration evaluations in Hyper-parameter tuning. A detailed pipeline figure of efficient hyper-parameter tuning using subset based training for faster configuration evaluations can be seen below:

<p align="center">
    <br>
        <img src="https://github.com/decile-team/cords/blob/main/docs/source/imgs/hpo_pipeline.png" height="400" width="700"/>
    </br>
</p>

We can use any existing data subset selection strategy in CORDS along with existing hyperparameter search and scheduling algorithms currently. 
We currently use [Ray-Tune](https://docs.ray.io/en/latest/tune/index.html) library for hyper-parameter tuning and search algorithms.

Please find the tutorial notebook explaining the usage of CORDS subset selections strategies for Efficient Hyper-parameter optimization in the following [notebook](https://github.com/decile-team/cords/blob/main/examples/HPO/image_classification/python_notebooks/CORDS_SL_CIFAR10_HPO_ASHA_Example.ipynb)

## Speedups achieved using CORDS
To achieve significantly faster speedups, one can use the subset selection data loaders from CORDS while keeping the training algorithm the same. Look at the speedups one can achieve using the subset selection data loaders from CORDS below:

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

## Tutorials
We have added example python code and tutorial notebooks under the examples folder. See [this link](https://github.com/decile-team/cords/tree/main/examples)

## Documentation

The documentation for the latest version of CORDS can always be found [here](https://cords.readthedocs.io/en/latest/).

## Mailing List
To receive updates about CORDS and to be a part of the community, join the Decile_CORDS_Dev group.
```
https://groups.google.com/forum/#!forum/Decile_CORDS_Dev/join 
```

## Acknowledgment
This library takes inspiration, builds upon, and uses pieces of code from several open source codebases. These include [Teppei Suzuki's consistency based SSL repository](https://github.com/perrying/pytorch-consistency-regularization) and [Richard Liaw's Tune repository](https://github.com/ray-project/ray/tree/master/python/ray/tune). Also, CORDS uses [submodlib](https://github.com/decile-team/submodlib) for submodular optimization.

## Team
DISTIL is created and maintained by [Krishnateja Killamsetty](https://krishnatejakillamsetty.me/), Dheeraj N Bhat, [Rishabh Iyer](https://www.rishiyer.com), and [Ganesh Ramakrishnan](https://www.cse.iitb.ac.in/~ganesh/). We look forward to have CORDS more community driven. Please use it and contribute to it for your efficient learning research, and feel free to use it for your commercial projects. We will add the major contributors here.

## Resources

[Blog Articles](https://decile-research.medium.com/)


## Publications

[1] Schwartz, Roy, et al. "Green ai." arXiv preprint arXiv:1907.10597 (2019).

[2] Strubell, Emma, Ananya Ganesh, and Andrew McCallum. "Energy and policy considerations for deep learning in NLP." In ACL 2019.

[3] Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, and Rishabh Iyer, [GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning](https://arxiv.org/abs/2012.10630), 35th AAAI Conference on Artificial Intelligence, AAAI 2021

[4] Krishnateja Killamsetty, Durga Sivasubramanian, Abir De, Ganesh Ramakrishnan, Baharan Mirzasoleiman, Rishabh Iyer, [Grad-Match: A Gradient Matching based Data Selection Framework for Efficient Learning](https://arxiv.org/abs/2103.00123), 2021

[5] Baharan Mirzasoleiman, Jeff Bilmes, and Jure Leskovec. [Coresets for Data-efficient Training of Machine Learning Models](https://arxiv.org/abs/1906.01827). In International Conference on Machine Learning (ICML), July 2020

[6] Kai Wei, Rishabh Iyer, Jeff Bilmes, [Submodularity in Data Subset Selection and Active Learning](http://proceedings.mlr.press/v37/wei15-supp.pdf), International Conference on Machine Learning (ICML) 2015

[7] Vishal Kaushal, Rishabh Iyer, Suraj Kothiwade, Rohan Mahadev, Khoshrav Doctor, and Ganesh Ramakrishnan, [Learning From Less Data: A Unified Data Subset Selection and Active Learning Framework for Computer Vision](https://arxiv.org/abs/1901.01151), 7th IEEE Winter Conference on Applications of Computer Vision (WACV), 2019 Hawaii, USA

[8] Wei, Kai, et al. [Submodular subset selection for large-scale speech training data](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.496.6287&rep=rep1&type=pdf), 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2014.

