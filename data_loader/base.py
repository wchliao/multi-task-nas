import numpy as np
import torch


class CustomDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class BaseDataLoader:
    def __init__(self, batch_size=1, type='train', shuffle=True, drop_last=False):
        pass

    def get_loader(self, task, prob):
        raise NotImplementedError

    @property
    def image_size(self):
        raise NotImplementedError

    @property
    def num_channels(self):
        raise NotImplementedError

    @property
    def num_classes(self):
        raise NotImplementedError


class MultiTaskDataLoader:
    def __init__(self, dataloaders, prob=None):
        self.dataloaders = dataloaders
        self.iters = [iter(loader) for loader in self.dataloaders]

        if prob is None:
            self.prob = np.ones(len(self.dataloaders)) / len(self.dataloaders)
        else:
            self.prob = prob

        self.size = sum([len(d) for d in self.dataloaders])
        self.step = 0


    def __iter__(self):
        return self


    def __next__(self):
        if self.step >= self.size:
            self.step = 0
            raise StopIteration

        task = np.random.choice(list(range(len(self.dataloaders))), p=self.prob)

        try:
            data, labels = self.iters[task].__next__()
        except StopIteration:
            self.iters[task] = iter(self.dataloaders[task])
            data, labels = self.iters[task].__next__()

        self.step += 1

        return data, labels, task