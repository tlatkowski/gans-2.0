import argparse

import utils


def run_experiment(dataset, gan_type):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_type',
                        required=True,
                        help='The path to training images',
                        choices=utils.dataset_type_values())
    
    parser.add_argument('--gan_type',
                        required=True,
                        help='The path to training images',
                        choices=utils.model_type_values())
    
    args = parser.parse_args()
    
    run_experiment(argparse.dataset_type, argparse.gan_type)
