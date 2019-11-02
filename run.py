import argparse

from utils import config
from datasets import dataset_factory
from models import model_factories


def run_experiment(input_args):
    gan_type = input_args.problem_type.split('_')[0]
    problem_type = input_args.problem_type
    problem_params = config.read_config(problem_type)
    dataset = dataset_factory.get_dataset(problem_params, problem_type)
    gan_model = model_factories.gan_model_factory(problem_params, gan_type, input_args)
    gan_model.fit(dataset)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--problem_type',
                        required=True,
                        help='The problem type',
                        choices=dataset_factory.dataset_type_values())
    
    parser.add_argument('-continue_training',
                        action='store_true',
                        help='If set the training process will be continued')
    
    args = parser.parse_args()
    
    run_experiment(args)


if __name__ == '__main__':
    main()
