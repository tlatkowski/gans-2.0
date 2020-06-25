import argparse

from gans.datasets import dataset_factory
from gans.datasets import problem_type
from gans.models import model_factories
from gans.utils import config
from gans.utils import logging

logger = logging.get_logger(__name__)


def run_experiment(input_args):
    problem_type = input_args.problem_type
    logger.info(f'Starting pipeline for {problem_type}...')
    gan_type = problem_type.split('_')[0]
    problem_params = config.read_config(problem_type)
    logger.info(f'Loaded parameters: \n {problem_params}')
    dataset = dataset_factory.get_dataset(problem_params, problem_type)
    logger.info(f'Loaded dataset: {dataset}')
    gan_model = model_factories.gan_model_factory(problem_params, gan_type, input_args)
    logger.info(f'Built GAN model: {gan_model}')
    logger.info('Model training...')
    gan_model.fit(dataset)
    logger.info('Training finished.')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--problem_type',
        required=True,
        help='The problem type',
        choices=problem_type.dataset_type_values(),
    )

    parser.add_argument(
        '-continue_training',
        action='store_true',
        help='If set the training process will be continued',
    )

    args = parser.parse_args()

    run_experiment(args)


if __name__ == '__main__':
    main()
