import yaml
from easydict import EasyDict as edict


def read_config(problem_type):
    with open('config/{}.yml'.format(problem_type.lower())) as f:
        config = edict(yaml.load(f))
        return config
