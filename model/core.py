import numpy as np
import torch.nn as nn
from namedtuple import ShareLayer


class _InputLayer(nn.Module):
    def __init__(self, layers):
        super(_InputLayer, self).__init__()

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)

        return x


class _OutputLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(_OutputLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        x = self.layers(x)

        return x


class _CoreModel(nn.Module):
    def __init__(self, layers, architecture, image_size, num_classes):
        super(_CoreModel, self).__init__()

        self.initial_layer = _InputLayer(layers)

        final_image_size = image_size // np.prod([args.stride for args in architecture])
        out_channels = architecture[-1].num_channels
        input_size = final_image_size * final_image_size * out_channels
        self.output_layer = _OutputLayer(input_size=input_size,
                                         output_size=num_classes)

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.output_layer(x)

        return x


def SingleTaskCoreModel(layers, architecture, task_info):
    layers = [ShareLayer(layer=layer, share=[0]) for layer in layers]
    _layers = _build_layers(layers=layers,
                            architecture=architecture,
                            num_tasks=1,
                            num_channels=task_info.num_channels
                            )
    return _CoreModel(_layers[0], architecture, task_info.image_size, task_info.num_classes)


def MultiTaskCoreModel(layers, architecture, task_info):
    _layers = _build_layers(layers=layers,
                            architecture=architecture,
                            num_tasks=task_info.num_tasks,
                            num_channels=task_info.num_channels
                            )
    return [_CoreModel(l, architecture, task_info.image_size, task_info.num_classes[i]) for i, l in enumerate(_layers)]


def _build_layers(layers, architecture, num_channels, num_tasks):
    models = [[] for _ in range(num_tasks)]
    in_channels = num_channels

    for layer, args in zip(layers, architecture):
        out_channels = args.num_channels
        kernel_size = layer.layer.kernel_size
        padding = (kernel_size - 1) // 2
        stride = args.stride

        if layer.layer.type == 'conv':
            shared_layer = nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     stride=stride
                                     )

            for model, share in zip(models, layer.share):
                if share:
                    model.append(shared_layer)
                else:
                    _layer = nn.Conv2d(in_channels,
                                       out_channels,
                                       kernel_size=kernel_size,
                                       padding=padding,
                                       stride=stride
                                       )
                    model.append(_layer)

        elif layer.layer.type == 'depthwise-conv':
            shared_layer1 = nn.Conv2d(in_channels,
                                      in_channels,
                                      kernel_size=kernel_size,
                                      padding=padding,
                                      stride=stride,
                                      groups=in_channels
                                      )
            shared_layer2 = nn.Conv2d(in_channels,
                                      out_channels,
                                      kernel_size=1,
                                      padding=0,
                                      stride=1
                                      )

            for model, share in zip(models, layer.share):
                if share:
                    model.append(shared_layer1)
                    model.append(shared_layer2)
                else:
                    _layer1 = nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        stride=stride,
                                        groups=in_channels
                                        )
                    _layer2 = nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=1,
                                        padding=0,
                                        stride=1
                                        )
                    model.append(_layer1)
                    model.append(_layer2)


        elif layer.layer.type == 'avg-pool':
            shared_layer = nn.AvgPool2d(kernel_size=kernel_size,
                                        padding=padding,
                                        stride=stride
                                        )

            for model, share in zip(models, layer.share):
                if share:
                    model.append(shared_layer)
                else:
                    _layer = nn.AvgPool2d(kernel_size=kernel_size,
                                          padding=padding,
                                          stride=stride
                                          )
                    model.append(_layer)

            if in_channels != out_channels:
                shared_layer = nn.Conv2d(in_channels,
                                         out_channels,
                                         kernel_size=1,
                                         padding=0,
                                         stride=1
                                         )

                for model, share in zip(models, layer.share):
                    if share:
                        model.append(shared_layer)
                    else:
                        _layer = nn.Conv2d(in_channels,
                                           out_channels,
                                           kernel_size=1,
                                           padding=0,
                                           stride=1
                                           )
                        model.append(_layer)

        elif layer.layer.type == 'max-pool':
            shared_layer = nn.MaxPool2d(kernel_size=kernel_size,
                                        padding=padding,
                                        stride=stride
                                        )

            for model, share in zip(models, layer.share):
                if share:
                    model.append(shared_layer)
                else:
                    _layer = nn.MaxPool2d(kernel_size=kernel_size,
                                          padding=padding,
                                          stride=stride
                                          )
                    model.append(_layer)

            if in_channels != out_channels:
                shared_layer = nn.Conv2d(in_channels,
                                         out_channels,
                                         kernel_size=1,
                                         padding=0,
                                         stride=1
                                         )

                for model, share in zip(models, layer.share):
                    if share:
                        model.append(shared_layer)
                    else:
                        _layer = nn.Conv2d(in_channels,
                                           out_channels,
                                           kernel_size=1,
                                           padding=0,
                                           stride=1
                                           )
                        model.append(_layer)

        else:
            raise ValueError('Unknown layer type: {}'.format(layer.type))

        shared_batchnorm = nn.BatchNorm2d(out_channels)

        for model, share in zip(models, layer.share):
            if share:
                model.append(shared_batchnorm)
                model.append(nn.ReLU())
            else:
                model.append(nn.BatchNorm2d(out_channels))
                model.append(nn.ReLU())

        in_channels = out_channels

    return models