import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from random import shuffle
from collections import OrderedDict
import dataloaders.base
from dataloaders.datasetGen import SplitGen, PermutedGen
import agents


def run(args):
    # Prepare dataloaders
    train_dataset, val_dataset = dataloaders.base.__dict__[args.dataset](args.dataroot, args.train_aug)
    if args.n_permutation > 0:
        train_dataset_splits, val_dataset_splits, task_output_space = PermutedGen(train_dataset, val_dataset,
                                                                             args.n_permutation,
                                                                             remap_class=not args.no_class_remap)
    else:
        train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                          first_split_sz=args.first_split_size,
                                                                          other_split_sz=args.other_split_size,
                                                                          rand_split=args.rand_split,
                                                                          remap_class=not args.no_class_remap)

    task_names = sorted(list(task_output_space.keys()), key=int)
    if len(args.eps_val) == 1:
        args.eps_val = [args.eps_val[0]] * len(task_names)
    if len(args.eps_max) == 1:
        args.eps_max = [args.eps_max[0]] * len(task_names)
    if len(args.eps_epoch) == 1:
        args.eps_epoch = [args.eps_epoch[0]] * len(task_names)
    if len(args.kappa_epoch) == 1:
        args.kappa_epoch = [args.kappa_epoch[0]] * len(task_names)
    if len(args.schedule) == 1:
        args.schedule = [args.schedule[0]] * len(task_names)

    # Prepare the Agent (model)
    agent_config = {'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay,
                    'schedule': args.schedule,
                    'model_type': args.model_type, 'model_name': args.model_name,
                    'model_weights': args.model_weights,
                    'out_dim': {'All': args.force_out_dim} if args.force_out_dim > 0 else task_output_space,
                    'optimizer': args.optimizer,
                    'print_freq': args.print_freq, 'gpuid': args.gpuid,
                    'reg_coef': args.reg_coef,
                    'force_out_dim': args.force_out_dim,
                    'clipping': args.clipping,
                    'eps_per_model': args.eps_per_model,
                    'milestones': args.milestones}
    agent = agents.__dict__[args.agent_type].__dict__[args.agent_name](agent_config)
    print(agent.model)
    print('#parameter of model:', agent.count_parameter())

    # Decide split ordering
    print('Task order:', task_names)
    if args.rand_split_order:
        shuffle(task_names)
        print('Shuffled task order:', task_names)

    acc_table = OrderedDict()
    if args.offline_training:  # Non-incremental learning / offline_training / measure the upper-bound performance
        task_names = ['All']
        train_dataset_all = torch.utils.data.ConcatDataset(train_dataset_splits.values())
        val_dataset_all = torch.utils.data.ConcatDataset(val_dataset_splits.values())
        train_loader = DataLoader(train_dataset_all, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.workers)
        val_loader = DataLoader(val_dataset_all, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers)

        agent.learn_batch(train_loader, val_loader)

        acc_table['All'] = {}
        acc_table['All']['All'] = agent.validation(val_loader)

    else:  # Incremental learning
        # Feed data to agent and evaluate agent's performance
        for i in range(len(task_names)):
            train_name = task_names[i]
            agent.current_task = int(task_names[i])
            print('======================', train_name, '=======================')
            train_loader = DataLoader(train_dataset_splits[train_name], batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.workers)
            val_loader = DataLoader(val_dataset_splits[train_name], batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.workers)

            if args.incremental_class:
                agent.add_valid_output_dim(task_output_space[train_name])

            if args.eps_max:
                agent.eps_scheduler.set_end(args.eps_max[i])

            agent.kappa_scheduler.end = args.kappa_min
            iter_on_batch = len(train_loader)
            agent.kappa_scheduler.calc_coefficient(args.kappa_min-1, args.kappa_epoch[i], iter_on_batch)
            agent.eps_scheduler.calc_coefficient(args.eps_val[i], args.eps_epoch[i], iter_on_batch)
            agent.kappa_scheduler.current, agent.eps_scheduler.current = 1, 0

            if agent.multihead:
                agent.current_head = str(train_name)

            print(f"before batch eps: {agent.eps_scheduler.current}, kappa: {agent.kappa_scheduler.current}")
            agent.learn_batch(train_loader, val_loader)  # Learn
            print(f"after batch eps: {agent.eps_scheduler.current}, kappa: {agent.kappa_scheduler.current}")

            if args.clipping:
                agent.save_params()

            agent.model.print_eps(agent.current_head)
            agent.model.reset_importance()

            # Evaluate
            acc_table[train_name] = OrderedDict()
            for j in range(i+1):
                val_name = task_names[j]
                print('validation split name:', val_name)
                val_data = val_dataset_splits[val_name] if not args.eval_on_train_set else train_dataset_splits[val_name]
                val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
                acc_table[val_name][train_name] = agent.validation(val_loader)
                agent.validation_with_move_weights(val_loader)

            agent.tb.close()

    return acc_table, task_names


def get_args(argv):
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--model_type', type=str, default='mlp', help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_name', type=str, default='MLP', help="The name of actual model for the backbone")
    parser.add_argument('--force_out_dim', type=int, default=2, help="Set 0 to let the task decide the required output dimension")
    parser.add_argument('--agent_type', type=str, default='interval', help="The type (filename) of agent")
    parser.add_argument('--agent_name', type=str, default='IntervalNet', help="The class name of agent")
    parser.add_argument('--optimizer', type=str, default='SGD', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='MNIST', help="MNIST(default)|CIFAR10|CIFAR100")
    parser.add_argument('--n_permutation', type=int, default=0, help="Enable permuted tests when >0")
    parser.add_argument('--first_split_size', type=int, default=2)
    parser.add_argument('--other_split_size', type=int, default=2)
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
                        help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,6 ...] -> [0,1,2 ...]")
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                        help="Randomize the order of splits")
    parser.add_argument('--workers', type=int, default=3, help="#Thread for dataloader")
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--kappa_epoch', nargs="+", type=float, default=[1])
    parser.add_argument('--kappa_min', type=float, default=0.5)
    parser.add_argument('--eps_epoch', nargs="+", type=float, default=[1])
    parser.add_argument('--eps_max', nargs="+", type=float, default=[0])
    parser.add_argument('--milestones', nargs="+", type=float, default=[])
    parser.add_argument('--eps_val', nargs="+", type=float)
    parser.add_argument('--eps_per_model', dest='eps_per_model', default=False, action='store_true')
    parser.add_argument('--clipping', dest='clipping', default=False, action='store_true')
    parser.add_argument('--schedule', nargs="+", type=int, default=[2],
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")
    parser.add_argument('--print_freq', type=float, default=100, help="Print the log at every x iteration")
    parser.add_argument('--model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--reg_coef', nargs="+", type=float, default=[0.], help="The coefficient for regularization. Larger means less plasilicity. Give a list for hyperparameter search.")
    parser.add_argument('--eval_on_train_set', dest='eval_on_train_set', default=False, action='store_true',
                        help="Force the evaluation on train set")
    parser.add_argument('--offline_training', dest='offline_training', default=False, action='store_true',
                        help="Non-incremental learning by make all data available in one batch. For measuring the upperbound performance.")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--incremental_class', dest='incremental_class', default=False, action='store_true',
                        help="The number of output node in the single-headed model increases along with new categories.")
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = get_args(sys.argv[1:])

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid[0])
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    reg_coef_list = args.reg_coef
    avg_final_acc = {}

    # The for loops over hyper-paramerters or repeats
    for reg_coef in reg_coef_list:
        args.reg_coef = reg_coef
        avg_final_acc[reg_coef] = np.zeros(args.repeat)
        for r in range(args.repeat):

            # Run the experiment
            acc_table, task_names = run(args)

            # Calculate average performance across tasks
            # Customize this part for a different performance metric
            avg_acc_history = [0] * len(task_names)
            for i in range(len(task_names)):
                train_name = task_names[i]
                cls_acc_sum = 0
                for j in range(i + 1):
                    val_name = task_names[j]
                    cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_history[i] = cls_acc_sum / (i + 1)
                print('Task', train_name, 'average acc:', avg_acc_history[i])

            # Gather the final avg accuracy
            avg_final_acc[reg_coef][r] = avg_acc_history[-1]

            # Print the summary so far
            print('===Summary of experiment repeats:', r+1, '/', args.repeat, '===')
            print('The regularization coefficient:', args.reg_coef)
            print('The last avg acc of all repeats:', avg_final_acc[reg_coef])
            print('mean:', avg_final_acc[reg_coef].mean(), 'std:', avg_final_acc[reg_coef].std())

    for reg_coef, v in avg_final_acc.items():
        print('reg_coef:', reg_coef, 'mean:', avg_final_acc[reg_coef].mean(), 'std:', avg_final_acc[reg_coef].std())

    print(f"* kappa decrease from 1 to {args.kappa_min} in {args.kappa_epoch} epoch")
    print(f"* eps increase by {args.eps_val} every {args.eps_epoch} epoch")
    print(f"* maximal eps: {args.eps_max if args.eps_max else 'inf'}")
    print(f"* tasks were trained {args.schedule} epoch {'with' if args.clipping else 'without'} clipping")
