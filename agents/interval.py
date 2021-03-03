import torch
import torch.optim as opt
import torch.nn as nn
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
from interval.layers import LinearInterval
from interval.hyperparam_scheduler import LinearScheduler


class IntervalNet(nn.Module):
    def __init__(self, agent_config):
        """
        :param agent_config (dict): lr=float,momentum=float,weight_decay=float,
                                    schedule=[int],  # The last number in the list is the end of epoch
                                    model_type=str,model_name=str,out_dim={task:dim},model_weights=str
                                    force_single_head=bool
                                    print_freq=int
                                    gpuid=[int]
        """
        super(IntervalNet, self).__init__()
        # Use a void function to replace the print
        self.log = print
        self.config = agent_config
        # If out_dim is a dict, there is a list of tasks. The model will have a head for each task.
        # A convenience flag to indicate multi-head/task
        self.multihead = True if len(self.config['out_dim']) > 1 else False
        self.model = self.create_model()
        self.criterion_fn = nn.CrossEntropyLoss()
        self.kappa_scheduler = LinearScheduler(start=1, end=0.5)
        self.eps_scheduler = LinearScheduler(start=0, end=0.3)
        self.interval_training = False
        self.previous_params, self.previous_last = {}, {}
        self.clipping = 0

        if agent_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        self.init_optimizer()
        self.reset_optimizer = False
        self.valid_out_dim = 'ALL'
        # Default: 'ALL' means all output nodes are active
        # Set a interger here for the incremental class scenario

        t = agent_config['force_out_dim'] if agent_config['force_out_dim'] else 2
        self.C = [-torch.eye(t).cuda() for _ in range(t)]
        for y0 in range(t):
            self.C[y0][y0, :] += 1

    def init_optimizer(self):
        optimizer_arg = {'params': self.model.parameters(),
                         'lr': self.config['lr'],
                         'weight_decay': self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD', 'RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'

        self.optimizer = opt.__dict__[self.config['optimizer']](**optimizer_arg)
        self.scheduler = opt.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['schedule'],
                                                      gamma=0.1)

    def create_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']]()

        # Apply network surgery to the backbone
        # Create the heads for tasks (It can be single task or multi-task)
        n_feat = model.last.in_features

        # The output of the model will be a dict: {task_name1:output1, task_name2:output2 ...}
        # For a single-headed model the output will be {'All':output}
        model.last = nn.ModuleDict()
        for task, out_dim in cfg['out_dim'].items():
            model.last[task] = LinearInterval(n_feat, out_dim)

        # Redefine the task-dependent function
        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            return outputs

        # Replace the task-dependent function
        model.logits = MethodType(new_logits, model)
        # Load pre-trained weights
        if cfg['model_weights'] is not None:
            print('=> Load model weights:', cfg['model_weights'])
            model_state = torch.load(cfg['model_weights'],
                                     map_location=lambda storage, loc: storage)  # Load to CPU.
            model.load_state_dict(model_state)
            print('=> Load Done')
        return model

    def forward(self, x):
        return self.model.forward(x)

    def predict(self, inputs):
        self.model.eval()
        out = self.forward(inputs)
        for t in out.keys():
            out[t] = out[t].detach()
        return out

    def validation(self, dataloader):
        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = self.training
        self.eval()
        for i, (inputs, target, task) in enumerate(dataloader):
            if self.gpu:
                with torch.no_grad():
                    inputs = inputs.cuda()
                    target = target.cuda()
            output = self.predict(inputs)

            # Summarize the performance of all tasks, or 1 task, depends on dataloader.
            # Calculated by total number of data.
            acc = accumulate_acc(output, target, task, acc)

        self.train(orig_mode)

        self.log(' * Val Acc {acc.avg:.3f}, time {time:.2f}'.format(acc=acc, time=batch_timer.toc()))
        return acc.avg

    def _interval_based_bound(self, y0, idx, key):
        # requires last layer to be linear
        C = self.C[y0].t()
        cW = C @ (self.model.last[key].weight - self.eps_scheduler.current)
        cb = C @ self.model.last[key].bias
        l, u = self.model.bounds
        return (cW.clamp(min=0) @ l[idx].t() + cW.clamp(max=0) @ u[idx].t() + cb[:, None]).t()

    def criterion(self, preds, targets, tasks, **kwargs):
        # The inputs and targets could come from single task or a mix of tasks
        # The network always makes the predictions with all its heads
        # The criterion will match the head and task to calculate the loss.

        if self.multihead:
            fit_loss, robust_loss = 0, 0
            for t, t_preds in preds.items():
                inds = [i for i in range(len(tasks)) if tasks[i] == t]  # The index of inputs that matched specific task
                if len(inds) > 0:
                    t_preds = t_preds[inds]
                    t_target = targets[inds]
                    # restore the loss from average
                    fit_loss += self.criterion_fn(t_preds, t_target) * len(inds)

                    for y0 in range(len(self.C)):
                        if (t_target == y0).sum().item() > 0:
                            lower_bound = self._interval_based_bound(y0, t_target == y0, key=t)
                            robust_loss += nn.CrossEntropyLoss(reduction='sum')(-lower_bound,
                                                                                t_target[t_target == y0]) / len(inds)
                            # increment when true label is not winning
                            # robust_err += (lower_bound.min(dim=1)[0] < 0).sum().item()

            loss = self.kappa_scheduler.current * fit_loss + (1 - self.kappa_scheduler.current) * robust_loss
            loss /= len(targets)  # Average the total loss by the mini-batch size
        else:
            pred = preds['All']
            # (Not 'ALL') Mask out the outputs of unseen classes for incremental class scenario
            if isinstance(self.valid_out_dim, int):
                pred = preds['All'][:, :self.valid_out_dim]
            loss = self.criterion_fn(pred, targets) * self.kappa_scheduler.current

            robust_loss, robust_err = 0, 0
            for y0 in range(len(self.C)):
                if (targets == y0).sum().item() > 0:
                    lower_bound = self._interval_based_bound(y0, targets == y0, key="All")
                    # (Not 'ALL') Mask out the outputs of unseen classes for incremental class scenario
                    if isinstance(self.valid_out_dim, int):
                        lower_bound = lower_bound[:, :self.valid_out_dim]

                    robust_loss += nn.CrossEntropyLoss(reduction='sum')(-lower_bound,
                                                                        targets[targets == y0]) / targets.size(0)
                    # increment when true label is not winning
                    # robust_err += (lower_bound.min(dim=1)[0] < 0).sum().item()

            loss += (1 - self.kappa_scheduler.current) * robust_loss

        return loss

    def update_model(self, inputs, targets, tasks):
        out = self.forward(inputs)
        loss = self.criterion(out, targets, tasks)
        self.optimizer.zero_grad()
        loss.backward()
        if self.clipping:
            for n, p in self.model.named_parameters():
                if "weight" in n:
                    low, upp = self.previous_params[n]-self.clipping, self.previous_params[n]+self.clipping
                    torch.where(p.data < low, low, p.data)
                    torch.where(p.data > upp, upp, p.data)

            for key, layer in self.model.last.items():
                for n, p in layer.named_parameters():
                    if "weight" in n:
                        low, upp = self.previous_last[key]-self.clipping, self.previous_last[key]+self.clipping
                        torch.where(p.data < low, low, p.data)
                        torch.where(p.data > upp, upp, p.data)

        self.optimizer.step()
        self.scheduler.step()

        self.kappa_scheduler.step()
        self.eps_scheduler.step()
        self.model.set_eps(self.eps_scheduler.current)

        return loss.detach(), out

    def save_previous_task_param(self):
        # Save previous params
        self.previous_params = {n: p.clone().detach()
                                for n, p in self.model.named_parameters()
                                if p.requires_grad and "weight" in n}
        self.previous_last = {}
        for key, layer in self.model.last.items():
            for n, p in layer.named_parameters():
                if "weight" in n:
                    self.previous_last[key] = p.clone().detach()

    def learn_batch(self, train_loader, val_loader=None):
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()

        for epoch in range(self.config['schedule'][-1]):
            data_timer = Timer()
            batch_timer = Timer()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            acc = AverageMeter()

            # Config the model and optimizer
            self.log('Epoch:{0}'.format(epoch))
            self.model.train()
            for param_group in self.optimizer.param_groups:
                self.log('LR:', param_group['lr'])

            # Learning with mini-batch
            data_timer.tic()
            batch_timer.tic()

            for i, (inputs, target, task) in enumerate(train_loader):
                data_time.update(data_timer.toc())  # measure data loading time
                if self.gpu:
                    inputs = inputs.cuda()
                    target = target.cuda()

                loss, output = self.update_model(inputs, target, task)
                inputs = inputs.detach()
                target = target.detach()

                # measure accuracy and record loss
                acc = accumulate_acc(output, target, task, acc)
                losses.update(loss, inputs.size(0))

                batch_time.update(batch_timer.toc())  # measure elapsed time
                data_timer.toc()

            self.log(' * Train Acc {acc.avg:.3f}, Loss {loss.avg:.3f}'.format(loss=losses, acc=acc))

            # Evaluate the performance of current task
            if val_loader is not None:
                self.validation(val_loader)

    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        if self.valid_out_dim == 'ALL':
            self.valid_out_dim = 0  # Initialize it with zero
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())

    def save_model(self, filename):
        model_state = self.model.state_dict()
        if isinstance(self.model, torch.nn.DataParallel):
            # Get rid of 'module' before the name of states
            model_state = self.model.module.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        print('=> Saving model to:', filename)
        torch.save(model_state, filename + '.pth')
        print('=> Save Done')

    def cuda(self):
        # torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'],
                                               output_device=self.config['gpuid'][0])
        return self


def accumulate_acc(output, target, task, meter):
    if 'All' in output.keys():  # Single-headed model
        meter.update(accuracy(output['All'], target), len(target))
    else:  # outputs from multi-headed (multi-task) model
        for t, t_out in output.items():
            inds = [i for i in range(len(task)) if task[i] == t]  # The index of inputs that matched specific task
            if len(inds) > 0:
                t_out = t_out[inds]
                t_target = target[inds]
                meter.update(accuracy(t_out, t_target), len(inds))

    return meter
