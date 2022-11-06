# RANN research paper submission

## Repository overview

This repository contains an analysis of autoencoder network architectures to compress and accurately reconstruct high-dimensional images of handwritten digits.

### manuscript/ directory

Contains latex files for compiling an accompanying manuscript of findings.

### src/ directory

Contains Python code relevant to the project.

-   `data/`: contains MNIST_train and MNIST test datasets, sourced from [Yann Lecun's website](http://yann.lecun.com/exdb/mnist/).
-   `models/`: contains trained instances of autoencoder networks.
-   `autoencoder.py`: custom class implementation for autoencoder networks.
-   `batch_estimate.py`: a script to queue estimation of a batch of autoencoder networks with various specified hyperparameters.
-   `main.ipynb`: Jupyter Notebook to define and estimate models, and produce figures and tables for the accompanying manuscript.

### requirements.txt

List of packages used by code in this repository. To run the code in this repository, you should install dependencies with:

```
pip install -r requriements.txt
```
