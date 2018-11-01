from .base import BaseController
from model import MultiTaskModelFull
from configs.search_space import search_space_full


class MultiTaskControllerFull(BaseController):
    def __init__(self, architecture, task_info):
        super(MultiTaskControllerFull, self).__init__(build_model=MultiTaskModelFull,
                                                      architecture=architecture,
                                                      search_space=search_space_full,
                                                      task_info=task_info
                                                      )
