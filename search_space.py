import yaml
from namedtuple import ShareLayer

with open('configs/search_space.yaml', 'r') as f:
    configs = yaml.load(f)

search_space_separate = [ShareLayer(**layer, share=False) for layer in configs]
search_space_shared = [ShareLayer(**layer, share=True) for layer in configs]
search_space_full = search_space_separate + search_space_shared