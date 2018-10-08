import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from .base import BaseModel
from .core import MultiTaskCoreModel


class MultiTaskModelFull(BaseModel):
    def __init__(self, layers, architecture, task_info):
        super(MultiTaskModelFull, self).__init__(layers, architecture, task_info)
        self.num_tasks = task_info.num_tasks
        self.models = [model.to(self.device) for model in MultiTaskCoreModel(layers=layers, architecture=architecture, task_info=task_info)]


    def train(self, train_data, valid_data, num_epochs=20, save_history=False, path='saved_models/default/', verbose=False):
        for model in self.models:
            model.train()

        dataloader = train_data.get_loader()
        criterion = nn.CrossEntropyLoss()
        optimizers = [optim.SGD(model.parameters(), lr=0.1) for model in self.models]
        accuracy = []

        for epoch in range(num_epochs):
            for inputs, labels, task in dataloader:
                model = self.models[task]
                optimizer = optimizers[task]

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            accuracy.append(self.eval(valid_data))

            if verbose:
                print('[Epoch {}] Accuracy: {}'.format(epoch + 1, accuracy[-1]))

        if save_history:
            self._save_history(accuracy, path)

        return np.mean(accuracy[-1])


    def _save_history(self, history, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        for i, h in enumerate(zip(*history)):
            filename = os.path.join(path, 'history_class{}.json'.format(i))

            with open(filename, 'w') as f:
                json.dump(h, f)


    def eval(self, data):
        correct = [0 for _ in range(self.num_tasks)]
        total = [0 for _ in range(self.num_tasks)]

        with torch.no_grad():
            for t, model in enumerate(self.models):
                model.eval()

                for inputs, labels in data.get_loader(t):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, predict_labels = torch.max(outputs.detach(), 1)

                    total[t] += labels.size(0)
                    correct[t] += (predict_labels == labels).sum().item()

                model.train()

            return [c / t for c, t in zip(correct, total)]


    def save(self, path='saved_models/default/'):
        if not os.path.isdir(path):
            os.makedirs(path)

        for t, model in enumerate(self.models):
            filename = os.path.join(path, 'model{}'.format(t))
            torch.save(model.state_dict(), filename)


    def load(self, path='saved_models/default/'):
        if os.path.isdir(path):
            for t, model in enumerate(self.models):
                filename = os.path.join(path, 'model{}'.format(t))
                model.load_state_dict(torch.load(filename))
