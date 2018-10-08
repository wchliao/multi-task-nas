from model import MultiTaskModelFull
from configs.search_space import search_space_full as search_space
from .base import BaseRandomSearch


class MultiTaskRandomSearchFull(BaseRandomSearch):
    def __init__(self, architecture, task_info):
        super(MultiTaskRandomSearchFull, self).__init__(build_model=MultiTaskModelFull,
                                                        architecture=architecture,
                                                        search_space=search_space,
                                                        task_info=task_info
                                                        )