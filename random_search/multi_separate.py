from model import MultiTaskModelSeparate
from configs.search_space import search_space_separate as search_space
from .base import BaseRandomSearch


class MultiTaskRandomSearchSeparate(BaseRandomSearch):
    def __init__(self, architecture, task_info):
        super(MultiTaskRandomSearchSeparate, self).__init__(build_model=MultiTaskModelSeparate,
                                                            architecture=architecture,
                                                            search_space=search_space,
                                                            task_info=task_info
                                                            )