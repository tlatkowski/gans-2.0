![](https://img.shields.io/badge/Python-3.6.8-blue.svg) ![](https://img.shields.io/badge/TensorFlow-2.0.0-blue.svg) ![](https://img.shields.io/badge/License-MIT-blue.svg) [![Build Status](https://travis-ci.com/tlatkowski/gans-2.0.svg?branch=cifar10-model)](https://travis-ci.com/tlatkowski/gans-2.0)

# GANs 2.0: Generative Adversarial Networks in TensorFlow 2.0
Vanilla GAN (MNIST)            |  Conditional GAN (MNIST)
:-------------------------:|:-------------------------:
![vanilla_mnist](./pics/vanilla_mnist.gif)  |  ![conditional_mnist](./pics/conditional_mnist.gif)


Vanilla GAN (FASHION_MNIST)            |  Conditional GAN (FASHION_MNIST)
:-------------------------:|:-------------------------:
![vanilla_fashion_mnist](./pics/vanilla_fashion_mnist.gif)  |  ![conditional_fashion_mnist](./pics/conditional_fashion_mnist.gif)


Vanilla GAN (CIFAR10)            |  Conditional GAN (CIFAR10)
:-------------------------:|:-------------------------:
![vanilla_cifar10](./pics/vanilla_cifar10.gif)  |  ![conditional_fashion_mnist](./pics/conditional_fashion_mnist.gif)

## Installation

In order to install all required dependencies, execute the following command:

in case of **CPU** usage:
```bash
./bin/install-cpu.sh
```
in case of **GPU** usage:
```bash
./bin/install-gpu.sh
```

The current version of pipeline supports the following datasets:
 * MNIST
 * FASHION MNIST
 * CIFAR10 
 
The set of supported Generative Adversarial Networks:
 * Vanilla GAN
 * Conditional GAN
 * Wasserstein GAN (TBD)

## Running pipeline

#### Model training from scratch
In order to run GAN training on a specific problem you have to provide one mandatory parameter:
 * **--problem_type**
 
The below command runs Vanilla GAN on MNIST dataset:
```bash
python run.py --problem_type VANILLA_MNIST
```

#### Restoring model training

In order to restore training from the latest checkpoint use **-continue_training** input argument, e.g.
```bash
python run.py --problem_type VANILLA_MNIST -continue_training
```
#### GAN problem configuration

The GAN problem configurations are placed at **./config**. If you want you can tweak those parameters, below is the example of VANILLA_MNIST problem configuration:
```json
{
  "batch_size": 256,
  "buffer_size": 60000,
  "num_epochs": 100,
  "hidden_size": 100,
  "img_height": 28,
  "img_width": 28,
  "num_channels": 1,
  "checkpoint_step": 10
} 
```
 
## Monitoring model training
In order to visualize a training process (loss values, generated outputs) run the following command in the project directory:
```bash
tensorboard --logdir outputs
```
To follow the training process go to the default browser and type the following address **http://your-workstation-name:6006/** 

The below picture presents the TensorBoard view lunched for two experiments: **VANILLA_MNIST** and **VANILLA_FASION_MNIST**.
![](./pics/tensorboard.png) 
## References
1. Deep Convolutional Generative Adversarial Network Tutorial in TensorFlow (https://www.tensorflow.org/beta/tutorials/generative/dcgan)

