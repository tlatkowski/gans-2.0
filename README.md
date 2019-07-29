![](https://img.shields.io/badge/Python-3.6-blue.svg) ![](https://img.shields.io/badge/TensorFlow-2.0.0-blue.svg) ![](https://img.shields.io/badge/License-MIT-blue.svg)

# Set of basic Generative Adversarial Networks experiments

## Installation

In order to install all required dependencies, execute the following command:

in case of CPU usage:
```bash
./bin/install-cpu.sh
```

in case of GPU usage:
```bash
./bin/install-gpu.sh
```

Current version of pipeline supports the following datasets:
 * MNIST
 * FASHION MNIST
 * CIFAR10 

## Running pipeline
In order to run GAN training on specific problem you have to provide two mandatory parameters:
 * **--dataset_type**
 * **--gan_type**
 
```bash
python run.py --dataset_type MNIST --gan_type VANILLA_GAN
```