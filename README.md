# CORDS: COResets and Data Subset selection
Reduce end to end training time from days to hours and hours to minutes using coresets and data selection. CORDS implements a number of state of the art data subset selection algorithms and coreset algorithms. Some of the algorithms currently implemented with CORDS include:

- GLISTER [1]
- GradMatchOMP [2]
- CRAIG [2,3]
- SubmodualrSelection [4,5,6]
  - Facility Location
  - Feature Based Functions
  - Coverage
  - Diversity
- RandomSelection

Publications:

[1] Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, and Rishabh Iyer, GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning, 35th AAAI Conference on Artificial Intelligence, AAAI 2021
[2] S Durga, Krishnateja Killamsetty, Abir De, Ganesh Ramakrishnan, Baharan Mirzasoleiman, Rishabh Iyer, Grad-Match: A Gradient Matching based Data Selection Framework for Efficient Learning
[3] Baharan Mirzasoleiman, Jeff Bilmes, and Jure Leskovec. Coresets for Data-efficient Training of Machine Learning Models. In International Conference on Machine Learning (ICML), July 2020.
[4] Kai Wei, Rishabh Iyer, Jeff Bilmes, Submodularity in data subset selection and active learning, International Conference on Machine Learning (ICML) 2015: 
[5] Vishal Kaushal, Rishabh Iyer, Suraj Kothiwade, Rohan Mahadev, Khoshrav Doctor, and Ganesh Ramakrishnan, Learning From Less Data: A Unified Data Subset Selection and Active Learning Framework for Computer Vision, 7th IEEE Winter Conference on Applications of Computer Vision (WACV), 2019 Hawaii, USA
[6] Wei, Kai, et al. "Submodular subset selection for large-scale speech training data." 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2014.

## Installation
The latest version of  CORDS package can be installed using the following command:

```python
pip install -i https://test.pypi.org/simple/ cords
```
---
**NOTE:**
  Please make sure to enter the space between simple/ and cords in the above command while installing CORDS package
---

## Package Requirements
1) "numpy >= 1.14.2",
2) "scipy >= 1.0.0",
3) "numba >= 0.43.0",
4) "tqdm >= 4.24.0",
5) "torch >= 1.4.0",
6) "apricot-select >= 0.6.0"
