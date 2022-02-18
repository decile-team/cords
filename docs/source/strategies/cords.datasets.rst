Datasets
====================
We have added functionality to load several existing datasets for both supervised and semi-supervised learning settings. 
Use the gen_dataset function in the file cords.utils.datasets.SL.builder for loading the dataset in the supervised learning
setting. Similarly, use the gen_dataset function in the file cords.utils.datasets.SSL.builder for loading the dataset in the
semi-supervised learning setting.

In the Supervised learning setting, below given is a list of datasets supported:
 - dna
 - boston
 - adult
 - connect_4
 - letter
 - satimage
 - svmguide1
 - usps
 - ijcnn1
 - sklearn-digits
 - covertype
 - census
 - mnist
 - fashion-mnist
 - cifar10
 - cifar100
 - svhn
 - kmnist
 - stl10
 - emnist
 - celeba

In the Semi-supervised learning setting, below given is a list of datasets supported:
 - svhn
 - stl10
 - cifar10
 - cifar100
 - cifarOOD
 - mnistOOD
 - cifarImbalance
 
.. toctree::
   :maxdepth: 5
   
   cords.datasets.SL.builder
   cords.datasets.SSL.builder