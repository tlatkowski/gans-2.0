import argparse
import configparser

from easydict import EasyDict as edict

import utils


def read_config(specific):
    config = configparser.ConfigParser()
    config.read('config/{}.ini'.format(specific.to_lower()))
    params = edict(config._sections)
    return params


def run_experiment(dataset_type, gan_type):
    # input_params = edict({'batch_size': 4, 'buffer_size': 60000})
    input_params = read_config(dataset_type)
    dataset = utils.dataset_factory(input_params, dataset_type)
    gan_model = utils.model_factory(input_params, gan_type)
    gan_model.fit(dataset)


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
    
    run_experiment(args.dataset_type, args.gan_type)
