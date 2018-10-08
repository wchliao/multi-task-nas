import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np
from .base import BaseModel


class SingleTaskModel(BaseModel):
    def __init__(self, layers, architecture, task_info):
        super(SingleTaskModel, self).__init__(layers, architecture, task_info)
        self.model = CoreModel(layers=layers, architecture=architecture, task_info=task_info).to(self.device)


    def train(self, train_data, valid_data, num_epochs=20, save_history=False, path='saved_models/default/', verbose=False):
        self.model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        accuracy = []

        for epoch in range(num_epochs):
            scheduler.step()
            for inputs, labels in train_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            accuracy.append(self.eval(valid_data))

            if verbose:
                print('[Epoch {}] Accuracy: {}'.format(epoch+1, accuracy[-1]))

        if save_history:
            self._save_history(accuracy, path)

        return accuracy[-1]


    def _save_history(self, history, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        filename = os.path.join(path, 'history.json')

        with open(filename, 'w') as f:
            json.dump(history, f)


    def eval(self, data):
        correct = 0
        total = 0

        with torch.no_grad():
            self.model.eval()

            for inputs, labels in data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predict_labels = torch.max(outputs.detach(), 1)

                total += labels.size(0)
                correct += (predict_labels == labels).sum().item()

            self.model.train()

            return correct / total


    def save(self, save_path='saved_models/default/'):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        filename = os.path.join(save_path, 'model')

        torch.save(self.model.state_dict(), filename)


    def load(self, save_path='saved_models/default/'):
        if os.path.isdir(save_path):
            filename = os.path.join(save_path, 'model')
            self.model.load_state_dict(torch.load(filename))


class InputLayer(nn.Module):
    def __init__(self, layers):
        super(InputLayer, self).__init__()

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)

        return x


class OutputLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(OutputLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        x = self.layers(x)

        return x


class CoreModel(nn.Module):
    def __init__(self, layers, architecture, task_info):
        super(CoreModel, self).__init__()

        _layers = self._build_layers(layers, architecture, task_info.num_channels)
        self.initial_layer = InputLayer(_layers)

        image_size = task_info.image_size // np.prod([args.stride for args in architecture])
        out_channels = architecture[-1].num_channels
        input_size = image_size * image_size * out_channels
        self.output_layer = OutputLayer(input_size=input_size,
                                         output_size=task_info.num_classes)


    def forward(self, x):
        x = self.initial_layer(x)
        x = self.output_layer(x)

        return x


    def _build_layers(self, layers, architecture, num_channels):
        _layers = []
        in_channels = num_channels

        for layer, args in zip(layers, architecture):
            out_channels = args.num_channels
            kernel_size = layer.kernel_size
            stride = args.stride

            if layer.type == 'conv':
                _layer = nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=kernel_size,
                                   padding=(kernel_size - 1) // 2,
                                   stride=stride
                                   )
                _layers.append(_layer)

            elif layer.type == 'depthwise-conv':
                _layer = nn.Conv2d(in_channels,
                                   in_channels,
                                   kernel_size=kernel_size,
                                   padding=(kernel_size - 1) // 2,
                                   stride=stride,
                                   groups=in_channels
                                   )
                _layers.append(_layer)

                _layer = nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=1,
                                   padding=0,
                                   stride=1
                                   )
                _layers.append(_layer)

            elif layer.type == 'avg-pool':
                _layer = nn.AvgPool2d(kernel_size=kernel_size,
                                      padding=(kernel_size-1) // 2,
                                      stride=stride
                                      )
                _layers.append(_layer)

                if in_channels != out_channels:
                    _layer = nn.Conv2d(in_channels,
                                       out_channels,
                                       kernel_size=1,
                                       padding=0,
                                       stride=1
                                       )
                    _layers.append(_layer)

            elif layer.type == 'max-pool':
                _layer = nn.MaxPool2d(kernel_size=kernel_size,
                                      padding=(kernel_size-1) // 2,
                                      stride=stride
                                      )
                _layers.append(_layer)

                if in_channels != out_channels:
                    _layer = nn.Conv2d(in_channels,
                                       out_channels,
                                       kernel_size=1,
                                       padding=0,
                                       stride=1
                                       )
                    _layers.append(_layer)
            else:
                raise ValueError('Unknown layer type: {}'.format(layer.type))

            _layers.append(nn.BatchNorm2d(out_channels))
            _layers.append(nn.ReLU())

            in_channels = out_channels

        return _layers
