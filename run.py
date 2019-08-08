import argparse

from utils import config
from utils import dataset_utils


# visualization.make_gif_from_images('outputs/CONDITIONAL_MNIST')

def run_experiment(input_args):
    problem_type = input_args.problem_type
    problem_params = config.read_config(problem_type)
    dataset = dataset_utils.problem_factory(problem_params, problem_type)
    gan_model = dataset_utils.model_factory(problem_params, input_args)
    gan_model.fit(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--problem_type',
                        required=True,
                        help='The problem type',
                        choices=dataset_utils.dataset_type_values())
    
    parser.add_argument('--gan_type',
                        required=True,
                        help='The GAN type',
                        choices=dataset_utils.model_type_values())
    
    parser.add_argument('-continue_training',
                        action='store_true',
                        help='If set the training process will be continued')
    
    args = parser.parse_args()
    
    run_experiment(args)
