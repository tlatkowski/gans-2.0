from setuptools import setup

setup(
    name='gans2',
    version='0.1',
    author='Tomasz Latkowski',
    description='GANs-2.0',
    long_description=(
        'Generative Adversarial Networks in TensorFlow 2.0.'
    ),
    long_description_content_type='text/markdown',
    install_requires=[
        'matplotlib==3.1.0',
        'easydict==1.9',
        'Pillow==7.1.0',
        'imageio==2.5.0',
        'tqdm==4.47.0',
        'tensorflow-addons~=0.6',
        'tensorflow-datasets==1.3.0',
        'pyyaml==5.1.2',
        'overrides==3.1.0',
    ],
    extras_require={
        'tensorflow': ['tensorflow>=2.2.0'],
    },
    packages=[
        'gans.datasets',
        'gans.layers',
        'gans.models',
        'gans.trainers',
        'gans.utils',
    ],
    zip_safe=True,
    python_requires='>=3.6',
)
