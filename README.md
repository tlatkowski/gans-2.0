![](https://img.shields.io/badge/Python-3.6-blue.svg) ![](https://img.shields.io/badge/TensorFlow-2.0.0-blue.svg) ![](https://img.shields.io/badge/License-MIT-blue.svg)

# GANs 2.0: a set of basic Generative Adversarial Networks implemented in TensorFlow 2.0
![cfashion_mnist](./pics/conditional_fashion_mnist.gif =128x128)
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
 
Supported Generative Adversarial Networks:
 * Vanilla GAN
 * Conditional GAN
 * Wasserstein GAN (TBD)

## Running pipeline

#### Model training from scratch
In order to run GAN training on specific problem you have to provide two mandatory parameters:
 * **--dataset_type**
 * **--gan_type**
 
```bash
python run.py --problem_type VANILLA_MNIST --gan_type VANILLA_GAN
```

#### Restoring model training

## Monitoring model training
In order to visualize a training process (loss values, generated outputs) run the following command in project directory:
```bash
tensorboard --logdir outputs
```
It will lunch TensorBoard. To follow the training process go to the following address **http://your-workstation-name:6006/** 

The below picture presents the TensorBoard view lunched after 3 experiments (CONDITIONAL_FASION_MNIST, CONDITIONAL_MNIST and CIFAR-10)

## References

