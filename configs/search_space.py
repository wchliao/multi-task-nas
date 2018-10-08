import yaml
from collections import namedtuple

with open('configs/search_space.yaml', 'r') as f:
    configs = yaml.load(f)

Layer = namedtuple('Layer', list(configs[0].keys()) + ['share'])

search_space_separate = [Layer(*layer.values(), share=False) for layer in configs]
search_space_shared = [Layer(*layer.values(), share=True) for layer in configs]
search_space_full = search_space_separate + search_space_shared