from .base import BaseController
from model import MultiTaskModelFull
from search_space import search_space_full


class MultiTaskControllerFullSearchSpace(BaseController):
    def __init__(self, architecture, task_info):
        super(MultiTaskControllerFullSearchSpace, self).__init__(build_model=MultiTaskModelFull,
                                                      architecture=architecture,
                                                      search_space=search_space_full,
                                                      task_info=task_info
                                                      )
