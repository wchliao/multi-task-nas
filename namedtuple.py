import yaml
from collections import namedtuple


# Named tuples for configurations

with open('configs/train.yaml', 'r') as f:
    _configs = yaml.load(f)

AgentConfigs = namedtuple('AgentConfigs', _configs['agent'].keys())
ModelConfigs = namedtuple('ModelConfigs', _configs['model'].keys())
Configs = namedtuple('Configs', ['agent', 'model'])

with open('configs/architecture.yaml', 'r') as f:
    _configs = yaml.load(f)

Layer = namedtuple('Layer', _configs[0].keys())

with open('configs/search_space.yaml', 'r') as f:
    _configs = yaml.load(f)

ShareLayer = namedtuple('ShareLayer', list(_configs[0].keys()) + ['share'])


# Others

TaskInfo = namedtuple('TaskInfo', ['image_size', 'num_classes', 'num_channels', 'num_tasks'])
