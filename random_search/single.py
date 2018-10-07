import numpy as np
import os
import json
import pickle
from .base import BaseRandomSearch
from .search_space import search_space_separate as search_space
from model import SingleTaskModel


class SingleRandomSearch(BaseRandomSearch):
    def __init__(self, architecture, task_info):
        super(SingleRandomSearch, self).__init__(architecture=architecture, task_info=task_info)
        self.search_space = search_space
        self.architecture = architecture
        self.task_info = task_info

        self.search_size = len(search_space)
        self.num_layers = len(architecture)

        self.sampled_architecture = []
        self.architecture_acc = []

        self.best_architecture = None
        self.best_acc = 0.


    def train(self,
              train_data,
              valid_data,
              test_data,
              configs,
              save_model=True,
              save_history=True,
              path='saved_models/default',
              verbose=False
              ):

        test_acc = []

        for epoch in range(configs.agent.num_epochs):
            layers = [self.search_space[0]]

            for _ in range(self.num_layers - 1):
                idx = np.random.randint(self.search_size)
                layers.append(self.search_space[idx])

            model = SingleTaskModel(layers, self.architecture, self.task_info)
            accuracy = model.train(train_data=train_data,
                                   valid_data=valid_data,
                                   num_epochs=configs.model.num_epochs,
                                   save_history=False,
                                   verbose=False
                                   )

            self.sampled_architecture.append(layers)
            self.architecture_acc.append(accuracy)

            if accuracy > self.best_acc:
                self.best_architecture = layers
                self.best_acc = accuracy
                test_acc.append(model.eval(test_data))
            else:
                test_acc.append(test_acc[-1])

            if verbose:
                print('[Epoch {}] Accuracy: {}'.format(epoch + 1, test_acc[-1]))

            if save_model:
                self.save(path)

            if save_history:
                self._save_history(test_acc, path)


    def _save_history(self, history, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        filename = os.path.join(path, 'history.json')

        with open(filename, 'w') as f:
            json.dump(history, f)


    def eval(self, train_data, test_data, configs):
        model = SingleTaskModel(self.best_architecture, self.architecture, self.task_info)
        accuracy = model.train(train_data=train_data,
                               valid_data=test_data,
                               num_epochs=configs.model.num_epochs,
                               save_history=False,
                               verbose=False
                               )

        return accuracy


    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        with open(os.path.join(path, 'sampled_architecture.pkl'), 'wb') as f:
            pickle.dump(self.sampled_architecture, f)
        with open(os.path.join(path, 'architecture_acc.pkl'), 'wb') as f:
            pickle.dump(self.architecture_acc, f)
        with open(os.path.join(path, 'best.pkl'), 'wb') as f:
            pickle.dump({'architecture': self.best_architecture, 'accuracy': self.best_acc}, f)


    def load(self, path):
        if os.path.isdir(path):
            with open(os.path.join(path, 'sampled_architecture.pkl'), 'rb') as f:
                self.sampled_architecture = pickle.load(f)
            with open(os.path.join(path, 'architecture_acc.pkl'), 'rb') as f:
                self.architecture_acc = pickle.load(f)
            with open(os.path.join(path, 'best.pkl'), 'rb') as f:
                best = pickle.load(f)
                self.best_architecture = best['architecture']
                self.best_acc = best['accuracy']
