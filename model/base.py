import torch


class BaseModel:
    def __init__(self, layers, architecture, task_info):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, train_data, test_data, num_epochs, save_history, save_path, verbose):
        raise NotImplementedError

    def eval(self, data):
        raise NotImplementedError

    def save(self, path):
        pass

    def load(self, path):
        pass