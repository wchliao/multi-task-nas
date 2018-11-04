import numpy as np
import os
import json
import pickle


class BaseRandomSearch:
    def __init__(self, build_model, architecture, search_space, task_info):
        self.build_model = build_model
        self.architecture = architecture
        self.search_space = search_space
        self.task_info = task_info

        self.search_size = len(search_space)
        self.num_layers = len(architecture)

        self.sampled_architecture = []
        self.architecture_acc = []

        self.best_architecture = None
        self.best_val_acc = 0.
        self.best_test_acc = []


    def train(self,
              train_data,
              valid_data,
              test_data,
              configs,
              save_model=True,
              path='saved_models/default',
              verbose=False
              ):

        for epoch in range(configs.agent.num_epochs):

            layers = None

            while layers is None or layers in self.sampled_architecture:
                layers = []

                for _ in range(self.num_layers):
                    idx = np.random.randint(self.search_size)
                    layers.append(self.search_space[idx])

            model = self.build_model(layers, self.architecture, self.task_info)
            accuracy = model.train(train_data=train_data,
                                   valid_data=valid_data,
                                   num_epochs=configs.model.num_epochs,
                                   learning_rate=configs.model.learning_rate,
                                   save_history=False,
                                   verbose=False
                                   )

            self.sampled_architecture.append(layers)
            self.architecture_acc.append(accuracy)

            if accuracy > self.best_val_acc:
                self.best_architecture = layers
                self.best_val_acc = accuracy
                self.best_test_acc.append(model.eval(test_data))
            else:
                self.best_test_acc.append(self.best_test_acc[-1])

            if verbose:
                print('[Epoch {}] Accuracy: {}'.format(epoch + 1, self.best_test_acc[-1]))

            if save_model:
                self.save(path)


    def eval(self, train_data, test_data, configs):
        model = self.build_model(self.best_architecture, self.architecture, self.task_info)
        accuracy = model.train(train_data=train_data,
                               valid_data=test_data,
                               num_epochs=configs.model.num_epochs,
                               learning_rate=configs.model.learning_rate,
                               save_history=False,
                               verbose=False
                               )

        return accuracy, self.best_architecture


    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        with open(os.path.join(path, 'sampled_architecture.pkl'), 'wb') as f:
            pickle.dump(self.sampled_architecture, f)
        with open(os.path.join(path, 'architecture_acc.json'), 'w') as f:
            json.dump(self.architecture_acc, f)
        with open(os.path.join(path, 'best.pkl'), 'wb') as f:
            pickle.dump({'architecture': self.best_architecture, 'accuracy': self.best_val_acc}, f)
        with open(os.path.join(path, 'history.json'), 'w') as f:
            json.dump(self.best_test_acc, f)


    def load(self, path):
        try:
            with open(os.path.join(path, 'sampled_architecture.pkl'), 'rb') as f:
                self.sampled_architecture = pickle.load(f)
            with open(os.path.join(path, 'architecture_acc.json'), 'r') as f:
                self.architecture_acc = json.load(f)

        except FileNotFoundError:
            self.sampled_architecture = []
            self.architecture_acc = []

        try:
            with open(os.path.join(path, 'best.pkl'), 'rb') as f:
                best = pickle.load(f)
                self.best_architecture = best['architecture']
                self.best_val_acc = best['accuracy']
            with open(os.path.join(path, 'history.json'), 'r') as f:
                self.best_test_acc = json.load(f)

        except FileNotFoundError:
            self.best_architecture = None
            self.best_val_acc = 0.
            self.best_test_acc = []
