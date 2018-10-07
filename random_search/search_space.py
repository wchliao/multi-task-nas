import yaml
from collections import namedtuple

with open('configs/search_space.yaml', 'r') as f:
    configs = yaml.load(f)

Layer = namedtuple('Layer', configs['separate'][0].keys())

search_space_separate = [Layer(*layer.values()) for layer in configs['separate']]
search_space_shared = [Layer(*layer.values()) for layer in configs['shared']]
search_space_full = search_space_separate + search_space_shared