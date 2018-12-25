from .base import BaseRandomSearch
from model import MultiTaskModel
from namedtuple import ShareLayer
from search_space import search_space as layers


class MultiTaskRandomSearchSeparate(BaseRandomSearch):
    def __init__(self, architecture, task_info):
        search_space = [ShareLayer(layer=layer, share=[0 for _ in range(task_info.num_tasks)]) for layer in layers]

        super(MultiTaskRandomSearchSeparate, self).__init__(build_model=MultiTaskModel,
                                                            architecture=architecture,
                                                            search_space=search_space,
                                                            task_info=task_info
                                                            )
