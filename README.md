# Neural Architecture Search for Multi-task Neural Networks

## Introduction

Train a neural architecture search (NAS) agent that can design multi-task models.

## Usage

### Train

```
python main.py --train
```

Arguments:

 * `--controller`: A flag used to decide whether to train a RL controller or not. If true, train a RL controller. Otherwise, random search a best model.
 * `--type`: (default: `3`)
   * `1`: Train a NAS agent for task *i* model.
   * `2`: Train a NAS agent for multi-task model using search space without share.
   * `3`: Train a NAS agent for multi-task model using full search space.
 * `--data`: (default: `1`)
   * `1`: CIFAR-100
 * `--task`: Task ID (for type `1`) (default: None)
 * `--load`: Load a pre-trained model before training. 
 * `--path`: Path (directory) that model and history are saved. (default: `'saved_models/default/'`)
 * `--save_model`: A flag used to decide whether to save model or not.
 * `--verbose`: A flag used to decide whether to demonstrate verbose messages or not.

### Evaluate

```
python main.py --eval
```

Arguments:

 * `--controller`: A flag used to decide whether to evaluate a RL controller or not. If true, evaluate a RL controller. Otherwise, evaluate a best random searched model.
 * `--type`: (default: `3`)
   * `1`: Evaluate a NAS agent for task *i* model.
   * `2`: Evaluate a NAS agent for multi-task model using search space without share.
   * `3`: Evaluate a NAS agent for multi-task model using full search space.
 * `--data`: (default: `1`)
   * `1`: CIFAR-100
 * `--task`: Task ID (for type `1`) (default: None)
 * `--path`: Path (directory) that model and history are saved. (default: `'saved_models/default/'`)
