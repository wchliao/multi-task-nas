from .base import BaseRandomSearch
from model import SingleTaskModel
from search_space import search_space


class SingleTaskRandomSearch(BaseRandomSearch):
    def __init__(self, architecture, task_info):
        super(SingleTaskRandomSearch, self).__init__(build_model=SingleTaskModel,
                                                     architecture=architecture,
                                                     search_space=search_space,
                                                     task_info=task_info
                                                     )
