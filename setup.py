from setuptools import setup
import setuptools
import os

long_description = "CORDS is COReset and Data Selection library for making machine learning time, energy, cost, and compute efficient. CORDS is built on top of PyTorch. Today, deep learning systems are extremely compute-intensive, with significant turnaround times, energy inefficiencies, higher costs, and resource requirements [7, 8]. CORDS is an effort to make deep learning more energy, cost, resource, and time-efficient while not sacrificing accuracy."

setup(
    name='cords',
    version='v0.0.4',
    author='Krishnateja Killamsetty, Dheeraj Bhat, Rishabh Iyer',
    author_email='krishnatejakillamsetty@gmail.com',
    #packages=['cords', 'cords/selectionstrategies', 'cords/utils'],
    url='https://github.com/decile-team/cords',
    license='LICENSE.txt',
    packages=setuptools.find_packages(),
    description='cords is a package for data subset selection for efficient and robust machine learning.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        "apricot-select>=0.6.0",
        "numba>=0.43.0",
        "scipy>=1.5.0",
        "scikit-learn",
        "torch>=1.8.0",
        "torchvision",
        "tqdm>=4.24.0",
        "sphinxcontrib-napoleon",
        "sphinxcontrib-bibtex",
        "sphinx-rtd-theme",
        "matplotlib",
        "numpy>=1.19.0",
        "torchvision>=0.10.1",
        "pillow>=8.4.0",
        "pandas>=1.1.0",
        "torchtext~=0.10.1",
        "scikit-image>=0.17.0",
        "pyyaml~=5.4.1",
        "dotmap~=1.3.24",
        "setuptools>=58.0.4",
        "ray[tune]",
        "ray[default]",
        "datasets"
            ],
)
