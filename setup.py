from setuptools import setup

setup(
    name='cords',
    version='v0.1',
    author='Krishnateja Killamsetty, Dheeraj Bhat, Rishabh Iyer',
    author_email='krishnatejakillamsetty@gmail.com',
    packages=['cords', 'cords/selectionstrategies', 'cords/utils'],
    url='http://pypi.python.org/pypi/apricot-select/',
    license='LICENSE.txt',
    #packages=setuptools.find_packages(),
    description='cords is a package for data subset selection for efficient and robust machine learning.',
    install_requires=[
        "numpy >= 1.14.2",
        "scipy >= 1.0.0",
        "numba >= 0.43.0",
        "tqdm >= 4.24.0",
        "torch >= 1.4.0"
    ],
)