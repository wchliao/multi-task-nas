class BaseRandomSearch:
    def __init__(self, architecture, task_info):
        pass

    def train(self, train_data, valid_data, test_data, configs, save_model, save_history, path, verbose):
        pass

    def eval(self, train_data, test_data, configs):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass
