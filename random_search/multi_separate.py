from .base import BaseRandomSearch
from model import MultiTaskModelSeparate
from search_space import search_space_separate


class MultiTaskRandomSearchSeparate(BaseRandomSearch):
    def __init__(self, architecture, task_info):
        super(MultiTaskRandomSearchSeparate, self).__init__(build_model=MultiTaskModelSeparate,
                                                            architecture=architecture,
                                                            search_space=search_space_separate,
                                                            task_info=task_info
                                                            )
