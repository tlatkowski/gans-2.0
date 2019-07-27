import json

from easydict import EasyDict as edict


def read_config(dataset_type):
    with open('config/{}.json'.format(dataset_type.lower())) as f:
        input_params = edict(json.load(f))
    return input_params
