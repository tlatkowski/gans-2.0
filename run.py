import argparse

from utils import config
from utils import dataset_utils


# visualization.make_gif_from_images('outputs/CONDITIONAL_MNIST')

def run_experiment(dataset_type, gan_type):
    input_params = config.read_config(dataset_type)
    dataset = dataset_utils.dataset_factory(input_params, dataset_type)
    gan_model = dataset_utils.model_factory(input_params, gan_type, dataset_type)
    gan_model.fit(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_type',
                        required=True,
                        help='The path to training images',
                        choices=dataset_utils.dataset_type_values())
    
    parser.add_argument('--gan_type',
                        required=True,
                        help='The path to training images',
                        choices=dataset_utils.model_type_values())
    
    args = parser.parse_args()
    
    run_experiment(args.dataset_type, args.gan_type)
