from .base import BaseController
from model import MultiTaskModelSeparate
from search_space import search_space_separate


class MultiTaskControllerSeparate(BaseController):
    def __init__(self, architecture, task_info):
        super(MultiTaskControllerSeparate, self).__init__(build_model=MultiTaskModelSeparate,
                                                          architecture=architecture,
                                                          search_space=search_space_separate,
                                                          task_info=task_info
                                                          )
