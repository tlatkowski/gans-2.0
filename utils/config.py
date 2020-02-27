import io
import json

import yaml
from easydict import EasyDict as edict


def read_config(dataset_type):
    with open('config/{}.json'.format(dataset_type.lower())) as f:
        input_params = edict(json.load(f))
    return input_params


def read_yml_config(path_to_config_file: str):
    with io.open(path_to_config_file) as file:
        config = yaml.load(file)
        return config
