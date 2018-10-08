import argparse
import yaml
from collections import namedtuple
from data_loader import CIFAR100Loader
from random_search import SingleRandomSearch


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--eval', action='store_true')

    parser.add_argument('--controller', action='store_true')
    parser.add_argument('--type', type=int, default=1, help='1: Single task experiment')
    parser.add_argument('--data', type=int, default=1, help='1: CIFAR-100')
    parser.add_argument('--task', type=int, default=None)
    parser.add_argument('--load', action='store_true')

    parser.add_argument('--path', type=str, default='saved_models/default/')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_history', action='store_true')

    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def train(args):
    configs = _load_configs()
    architecture = _load_architerture()

    if args.data == 1:
        train_data = CIFAR100Loader(batch_size=configs.model.batch_size, type='train', drop_last=True)
        valid_data = CIFAR100Loader(batch_size=configs.model.batch_size, type='valid', drop_last=False)
        test_data = CIFAR100Loader(batch_size=configs.model.batch_size, type='test', drop_last=False)
    else:
        raise ValueError('Unknown data ID: {}'.format(args.data))

    num_tasks = len(train_data.num_classes)

    if args.type == 1:
        assert args.task in list(range(num_tasks)), 'Unknown task: {}'.format(args.task)

        TaskInfo = namedtuple('TaskInfo', ['image_size', 'num_classes', 'num_channels'])
        task_info = TaskInfo(image_size=train_data.image_size,
                             num_classes=train_data.num_classes[args.task],
                             num_channels=train_data.num_channels)

        train_data = train_data.get_loader(args.task)
        valid_data = valid_data.get_loader(args.task)
        test_data = test_data.get_loader(args.task)

        if args.controller:
            raise NotImplementedError
        else:
            agent = SingleRandomSearch(architecture=architecture, task_info=task_info)
    else:
        raise ValueError('Unknown setting: {}'.format(args.setting))

    if args.load:
        agent.load(args.path)

    agent.train(train_data=train_data,
                valid_data=valid_data,
                test_data=test_data,
                configs=configs,
                save_model=args.save_model,
                path=args.path,
                verbose=args.verbose
                )

    if args.save_model:
        agent.save(args.path)


def evaluate(args):
    configs = _load_configs()
    architecture = _load_architerture()

    if args.data == 1:
        train_data = CIFAR100Loader(batch_size=configs.model.batch_size, type='train', drop_last=True)
        test_data = CIFAR100Loader(batch_size=configs.model.batch_size, type='test', drop_last=False)
    else:
        raise ValueError('Unknown data ID: {}'.format(args.data))

    num_tasks = len(train_data.num_classes)

    if args.type == 1:
        assert args.task in list(range(num_tasks)), 'Unknown task: {}'.format(args.task)

        TaskInfo = namedtuple('TaskInfo', ['image_size', 'num_classes', 'num_channels'])
        task_info = TaskInfo(image_size=train_data.image_size,
                             num_classes=train_data.num_classes[args.task],
                             num_channels=train_data.num_channels)

        train_data = train_data.get_loader(args.task)
        test_data = test_data.get_loader(args.task)

        if args.controller:
            raise NotImplementedError
        else:
            agent = SingleRandomSearch(architecture=architecture, task_info=task_info)
    else:
        raise ValueError('Unknown setting: {}'.format(args.setting))

    agent.load(args.path)
    accuracy = agent.eval(train_data=train_data,
                          test_data=test_data,
                          configs=configs
                          )

    print('Accuracy: {}'.format(accuracy))


def _load_configs():
    with open('configs/train.yaml', 'r') as f:
        configs = yaml.load(f)

    ControllerConfigs = namedtuple('ControllerConfigs', configs['agent'].keys())
    ModelConfigs = namedtuple('ModelConfigs', configs['model'].keys())
    Configs = namedtuple('Configs', ['agent', 'model'])

    agent_configs = ControllerConfigs(*configs['agent'].values())
    model_configs = ModelConfigs(*configs['model'].values())

    return Configs(agent=agent_configs, model=model_configs)


def _load_architerture():
    with open('configs/architecture.yaml', 'r') as f:
        configs = yaml.load(f)
    LayerArgument = namedtuple('LayerArgument', configs[0].keys())
    return [LayerArgument(*config.values()) for config in configs]


def main():
    args = parse_args()
    if args.train:
        train(args)
    elif args.eval:
        evaluate(args)
    else:
        print('No flag is assigned. Please assign either \'--train\' or \'--eval\'.')


if __name__ == '__main__':
    main()