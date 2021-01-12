# CORDS: COResets and Data Subset selection
Reduce end to end training time from days to hours and hours to minutes using coresets and data selection. CORDS implements a number of state of the art data subset selection algorithms and coreset algorithms. Some of the algorithms currently implemented with CORDS include:

- GLISTER [1]
- GradMatchOMP [2]
- GradMatchFixed [2] 
- CRAIG [2,3]
- SubmodularSelection [4,5,6]
  - Facility Location
  - Feature Based Functions
  - Coverage
  - Diversity
- RandomSelection

## Publications

[1](https://arxiv.org/abs/2012.10630) Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, and Rishabh Iyer, GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning, 35th AAAI Conference on Artificial Intelligence, AAAI 2021

[2] S Durga, Krishnateja Killamsetty, Abir De, Ganesh Ramakrishnan, Baharan Mirzasoleiman, Rishabh Iyer, Grad-Match: A Gradient Matching based Data Selection Framework for Efficient Learning

[3](https://arxiv.org/abs/1906.01827) Baharan Mirzasoleiman, Jeff Bilmes, and Jure Leskovec. Coresets for Data-efficient Training of Machine Learning Models. In International Conference on Machine Learning (ICML), July 2020.

[4](http://proceedings.mlr.press/v37/wei15-supp.pdf) Kai Wei, Rishabh Iyer, Jeff Bilmes, Submodularity in data subset selection and active learning, International Conference on Machine Learning (ICML) 2015: 

[5](https://arxiv.org/abs/1901.01151) Vishal Kaushal, Rishabh Iyer, Suraj Kothiwade, Rohan Mahadev, Khoshrav Doctor, and Ganesh Ramakrishnan, Learning From Less Data: A Unified Data Subset Selection and Active Learning Framework for Computer Vision, 7th IEEE Winter Conference on Applications of Computer Vision (WACV), 2019 Hawaii, USA

[6](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.496.6287&rep=rep1&type=pdf) Wei, Kai, et al. "Submodular subset selection for large-scale speech training data." 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2014.

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

---
**NOTE:**
  Please make sure to enter the space between simple/ and cords in the above command while installing CORDS package using PyPI
---

## Tutorials
The below link contains the jupyter notebook link for cifar10 timing analysis experiments

[CIFAR10 Notebook](https://colab.research.google.com/drive/1xT6sGmDGMz8XBDmOKs5cl1cipX0Ss1sh?usp=sharing)

## Package Requirements (this toolkit is built on top of the following packages)
1) "numpy >= 1.14.2",
2) "scipy >= 1.0.0",
3) "numba >= 0.43.0",
4) "tqdm >= 4.24.0",
5) "torch >= 1.4.0",
6) "apricot-select >= 0.6.0"
