import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import json
import pickle
from model import MultiTaskModelFull
from search_space import search_space_separate, search_space_shared


class MultiTaskControllerFull:
    def __init__(self, architecture, task_info):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.architecture = architecture
        self.search_space = [search_space_separate, search_space_shared]
        self.task_info = task_info

        self.num_layers = len(architecture)
        self.search_size = len(search_space_separate)

        self.controller = _Controller(search_size=self.search_size,
                                      num_outputs=self.num_layers,
                                      device=self.device
                                      )

        self.sampled_architecture = []
        self.architecture_acc_val = []
        self.architecture_acc_test = []
        self.history = []
        self.baseline = None


    def train(self,
              train_data,
              valid_data,
              test_data,
              configs,
              save_model=True,
              path='saved_models/default',
              verbose=False
              ):

        self.controller.train()

        optimizer = optim.Adam(self.controller.parameters(), lr=configs.agent.learning_rate)

        for epoch in range(configs.agent.num_epochs):
            actions, log_probs = self.controller.sample()
            layer_IDs = actions[0::2]
            share = actions[1::2]
            layers = [self.search_space[s][l] for l, s in zip(layer_IDs, share)]

            if layers in self.sampled_architecture:
                idx = self.sampled_architecture.index(layers)
                accuracy = self.architecture_acc_val[idx]
                test_acc = self.architecture_acc_test[idx]

            else:
                model = MultiTaskModelFull(layers, self.architecture, self.task_info)
                accuracy = model.train(train_data=train_data,
                                       valid_data=valid_data,
                                       num_epochs=configs.model.num_epochs,
                                       learning_rate=configs.model.learning_rate,
                                       save_history=False,
                                       verbose=False
                                       )
                test_acc = model.eval(test_data)

                self.sampled_architecture.append(layers)
                self.architecture_acc_val.append(accuracy)
                self.architecture_acc_test.append(test_acc)

            self.history.append(test_acc)

            if self.baseline is None:
                self.baseline = accuracy
            else:
                self.baseline = configs.agent.baseline_decay * self.baseline + (1 - configs.agent.baseline_decay) * accuracy

            advantage = accuracy - self.baseline
            loss = -log_probs * advantage
            loss = loss.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose:
                print('[Epoch {}] Accuracy: {}'.format(epoch + 1, self.history[-1]))

            if save_model:
                self.save(path)


    def eval(self, train_data, test_data, configs):
        with torch.no_grad():
            self.controller.eval()

            actions, _ = self.controller.sample(sample_best=True)
            layer_IDs = actions[0::2]
            share = actions[1::2]
            layers = [self.search_space[s][l] for l, s in zip(layer_IDs, share)]

        model = MultiTaskModelFull(layers, self.architecture, self.task_info)
        accuracy = model.train(train_data=train_data,
                               valid_data=test_data,
                               num_epochs=configs.model.num_epochs,
                               learning_rate=configs.model.learning_rate,
                               save_history=False,
                               verbose=False
                               )

        return accuracy, layers


    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        with open(os.path.join(path, 'sampled_architecture.pkl'), 'wb') as f:
            pickle.dump(self.sampled_architecture, f)
        with open(os.path.join(path, 'architecture_acc_val.json'), 'w') as f:
            json.dump(self.architecture_acc_val, f)
        with open(os.path.join(path, 'architecture_acc_test.json'), 'w') as f:
            json.dump(self.architecture_acc_test, f)
        with open(os.path.join(path, 'history.json'), 'w') as f:
            json.dump(self.history, f)
        with open(os.path.join(path, 'baseline.json'), 'w') as f:
            json.dump(self.baseline, f)

        torch.save(self.controller.state_dict(), os.path.join(path, 'controller'))


    def load(self, path):
        try:
            with open(os.path.join(path, 'sampled_architecture.pkl'), 'rb') as f:
                self.sampled_architecture = pickle.load(f)
            with open(os.path.join(path, 'architecture_acc_val.json'), 'r') as f:
                self.architecture_acc_val = json.load(f)
            with open(os.path.join(path, 'architecture_acc_test.json'), 'r') as f:
                self.architecture_acc_test = json.load(f)

        except FileNotFoundError:
            self.sampled_architecture = []
            self.architecture_acc_val = []
            self.architecture_acc_test = []

        try:
            with open(os.path.join(path, 'history.json'), 'r') as f:
                self.history = json.load(f)
            with open(os.path.join(path, 'baseline.json'), 'r') as f:
                self.baseline = json.load(f)

            self.controller.load_state_dict(torch.load(os.path.join(path, 'controller')))

        except FileNotFoundError:
            self.history = []
            self.baseline = None


class _Controller(nn.Module):
    def __init__(self, search_size, num_outputs, hidden_size=128, device=torch.device('cpu')):
        super(_Controller, self).__init__()

        layer_encoder = torch.nn.Embedding(search_size, hidden_size)
        share_encoder = torch.nn.Embedding(2, hidden_size)
        self.encoder = nn.ModuleList([layer_encoder, share_encoder])

        self.cell = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)

        self.decoder = []
        for _ in range(num_outputs):
            self.decoder.append(nn.Linear(hidden_size, search_size))
            self.decoder.append(nn.Linear(hidden_size, 2))
        self.decoder = nn.ModuleList(self.decoder)

        self.search_size = search_size
        self.num_outputs = num_outputs
        self.hidden_size = hidden_size
        self.device = device

        self.to(device)


    def forward(self, inputs, hidden, idx, is_embed=False):
        if is_embed:
            embed = inputs
        else:
            embed = self.encoder[(idx - 1) % 2](inputs)

        h, c = self.cell(embed, hidden)
        logits = self.decoder[idx](h)

        return logits, (h, c)


    def sample(self, sample_best=False):
        inputs = torch.zeros(1, self.hidden_size).to(self.device)
        hidden = (torch.zeros(1, self.hidden_size).to(self.device), torch.zeros(1, self.hidden_size).to(self.device))

        actions = []
        log_probs = []

        for idx in range(self.num_outputs * 2):
            logits, hidden = self.forward(inputs=inputs, hidden=hidden, idx=idx, is_embed=(idx == 0))
            prob = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1).squeeze()

            if sample_best:
                action = prob.argmax(dim=-1).squeeze().detach()
            else:
                action = prob.multinomial(num_samples=1).squeeze().detach()

            log_prob = log_prob[action]

            actions.append(action)
            log_probs.append(log_prob)

            inputs = action.view(1)

        return actions, torch.stack(log_probs)
