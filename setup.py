from setuptools import setup
import setuptools

setup(
    name='cords',
    version='0.1.0',
    author='Krishnateja Killamsetty, Dheeraj Bhat, Rishabh Iyer',
    author_email='krishnatejakillamsetty@gmail.com',
    #packages=['cords', 'cords/selectionstrategies', 'cords/utils'],
    url='https://github.com/decile-team/cords',
    license='LICENSE.txt',
    packages=setuptools.find_packages(),
    description='cords is a package for data subset selection for efficient and robust machine learning.',
    install_requires=[
        "numpy >= 1.14.2",
        "scipy >= 1.0.0",
        "numba >= 0.43.0",
        "tqdm >= 4.24.0",
        "torch >= 1.4.0",
        "apricot-select >= 0.6.0",
        "sphinxcontrib-napoleon",
        "sphinxcontrib-bibtex",
        "sphinx-rtd-theme",
        "scikit-learn",
        "torchvision >= 0.5.0",
        "matplotlib"
    ],
)
