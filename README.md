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
 * `--type`: (default: `4`)
   * `1`: Train a NAS agent for task *i* model.
   * `2`: Train a NAS agent for multi-task model using search space without shared components.
   * `3`: Train a NAS agent for multi-task model using search space with only shared components.
   * `4`: Train a NAS agent for multi-task model using full search space.
   * `5`: Train a NAS agent for multi-task model using controller that can control whether to share a layer among all tasks or not.
   * `6`: Train a NAS agent for multi-task model using controller that can control whether to share a layer among some tasks or not.
 * `--data`: (default: `1`)
   * `1`: CIFAR-100
 * `--task`: Task ID (for type `1`) (default: None) 
 * `--save`: A flag used to decide whether to save model or not.
 * `--load`: Load a pre-trained model before training.
 * `--path`: Path (directory) that model and history are saved. (default: `'saved_models/default/'`)
 * `--verbose`: A flag used to decide whether to demonstrate verbose messages or not.

### Evaluate

```
python main.py --eval
```

Arguments:

 * `--controller`: A flag used to decide whether to evaluate a RL controller or not. If true, evaluate a RL controller. Otherwise, evaluate a best random searched model.
 * `--type`: (default: `4`)
   * `1`: Evaluate a NAS agent for task *i* model.
   * `2`: Evaluate a NAS agent for multi-task model using search space without shared components.
   * `3`: Evaluate a NAS agent for multi-task model using search space with only shared components.
   * `4`: Evaluate a NAS agent for multi-task model using full search space.
   * `5`: Evaluate a NAS agent for multi-task model using controller that can control whether to share a layer among all tasks or not.
   * `6`: Evaluate a NAS agent for multi-task model using controller that can control whether to share a layer among some tasks or not.
 * `--data`: (default: `1`)
   * `1`: CIFAR-100
 * `--task`: Task ID (for type `1`) (default: None)
 * `--path`: Path (directory) that model and history are saved. (default: `'saved_models/default/'`)
