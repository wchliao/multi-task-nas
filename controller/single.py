from .base import BaseController
from model import SingleTaskModel
from search_space import search_space


class SingleTaskController(BaseController):
    def __init__(self, architecture, task_info):
        super(SingleTaskController, self).__init__(build_model=SingleTaskModel,
                                                   architecture=architecture,
                                                   search_space=search_space,
                                                   task_info=task_info
                                                   )
