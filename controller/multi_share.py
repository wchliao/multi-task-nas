from .base import BaseController
from model import MultiTaskModel
from namedtuple import ShareLayer
from search_space import search_space as layers


class MultiTaskControllerShare(BaseController):
    def __init__(self, architecture, task_info):
        search_space = [ShareLayer(layer=layer, share=[1 for _ in range(task_info.num_tasks)]) for layer in layers]

        super(MultiTaskControllerShare, self).__init__(build_model=MultiTaskModel,
                                                       architecture=architecture,
                                                       search_space=search_space,
                                                       task_info=task_info
                                                       )
