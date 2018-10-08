import torch.nn as nn
import torch.optim as optim
from .multi_full import MultiTaskModelFull


class MultiTaskModelSeparate(MultiTaskModelFull):
    def __init__(self, layers, architecture, task_info):
        super(MultiTaskModelSeparate, self).__init__(layers, architecture, task_info)


    def train(self, train_data, valid_data, num_epochs=20, save_history=False, path='saved_models/default/', verbose=False):
        for model in self.models:
            model.train()

        dataloaders = [train_data.get_loader(t) for t in range(self.num_tasks)]
        criterion = nn.CrossEntropyLoss()
        optimizers = [optim.SGD(model.parameters(), lr=0.1) for model in self.models]
        accuracy = []

        for epoch in range(num_epochs):
            for model, optimizer, dataloader in zip(self.models, optimizers, dataloaders):
                for inputs, labels in dataloader:
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

        return accuracy[-1]
