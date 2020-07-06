from setuptools import setup

with open('requirements/requirements.txt') as f:
    requirements = f.readlines()

setup(
    name='GANs-2.0',
    version='0.1',
    author='Tomasz Latkowski',
    description='GANs2.0',
    long_description=(
        'Generative Adversarial Networks in TensorFlow 2.0.'
    ),
    long_description_content_type='text/markdown',
    install_requires=requirements,
    extras_require={
        'tensorflow': ['tensorflow>=2.2.0'],
    },
    packages=[
        'gans.datasets',
        'gans.layers',
        'gans.models',
        'gans.trainers',
    ],
    zip_safe=True
)
'Generative Adversarial Networks in TensorFlow 2.0'
