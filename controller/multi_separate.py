from .base import BaseController
from model import MultiTaskModelSeparate
from namedtuple import ShareLayer
from search_space import search_space as layers


class MultiTaskControllerSeparate(BaseController):
    def __init__(self, architecture, task_info):
        search_space = [ShareLayer(layer=layer, share=[0 for _ in range(task_info.num_tasks)]) for layer in layers]

        super(MultiTaskControllerSeparate, self).__init__(build_model=MultiTaskModelSeparate,
                                                          architecture=architecture,
                                                          search_space=search_space,
                                                          task_info=task_info
                                                          )
