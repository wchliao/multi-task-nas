from .multi_full import MultiTaskModelFull


class MultiTaskModelSeparate(MultiTaskModelFull):
    def __init__(self, layers, architecture, task_info):
        super(MultiTaskModelSeparate, self).__init__(layers, architecture, task_info)
