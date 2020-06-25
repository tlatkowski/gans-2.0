from setuptools import setup

with open('requirements/requirements.txt') as f:
    requirements = f.readlines()

setup(
    name='GANs-2.0',
    version='0.1',
    author='Tomasz Latkowski',
    description='Generative Adversarial Networks in TensorFlow 2.0',
    long_description=open('README.md').read(),
    install_requires=requirements,
    packages=[
        'gans.datasets',
        'gans.layers',
        'gans.models',
        'gans.trainers',
    ],
    zip_safe=True
)
