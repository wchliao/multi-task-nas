from .base import BaseController
from model import MultiTaskModel
from namedtuple import ShareLayer
from search_space import search_space as layers


class MultiTaskControllerFullSearchSpace(BaseController):
    def __init__(self, architecture, task_info):
        search_space_separate = [ShareLayer(layer=layer, share=[0 for _ in range(task_info.num_tasks)]) for layer in layers]
        search_space_share = [ShareLayer(layer=layer, share=[1 for _ in range(task_info.num_tasks)]) for layer in layers]
        search_space = search_space_separate + search_space_share

        super(MultiTaskControllerFullSearchSpace, self).__init__(build_model=MultiTaskModel,
                                                                 architecture=architecture,
                                                                 search_space=search_space,
                                                                 task_info=task_info
                                                                 )
