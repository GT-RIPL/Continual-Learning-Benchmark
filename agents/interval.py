import torch
import torch.optim as opt
import torch.nn as nn
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
from interval.layers import LinearInterval, Conv2dInterval
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
        self.eps_scheduler = LinearScheduler(start=0)
        self.prev_weight, self.prev_eps = {}, {}
        self.clipping = self.config['clipping']
        self.current_head = "All"
        self.schedule_stack = []
        for s in self.config["schedule"][::-1]:
            self.schedule_stack.append(s)

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

        t = agent_config['force_out_dim'] if agent_config['force_out_dim'] else self.model.last["1"].out_features
        self.C = [-torch.eye(t).cuda() for _ in range(t)]
        for y0 in range(t):
            self.C[y0][y0, :] += 1

    def init_optimizer(self):
        optimizer_arg = {'params': (p for p in self.model.parameters() if p.requires_grad),
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
        # self.scheduler = opt.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['schedule'],
        #                                               gamma=0.1)

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

    def restore_weights(self):
        i = 0
        for c in self.model.children():
            if isinstance(c, nn.Sequential):
                for layer in c.children():
                    if isinstance(layer, (Conv2dInterval, LinearInterval)):
                        layer.weight.data = self.prev_weight[i].clone()
                        i += 1
            elif isinstance(c, nn.ModuleDict) and not self.multihead:
                c["All"].weight.data = self.prev_weight[i].clone()
                i += 1
            elif isinstance(c, (Conv2dInterval, LinearInterval)):
                c.weight.data = self.prev_weight[i].clone()
                i += 1

    def move_weights(self, sign):
        for c in self.model.children():
            if isinstance(c, nn.Sequential):
                for layer in c.children():
                    if isinstance(layer, (Conv2dInterval, LinearInterval)):
                        layer.weight.data += sign * layer.eps
            elif isinstance(c, nn.ModuleDict) and not self.multihead:
                c["All"].weight.data += sign * c["All"].eps
            elif isinstance(c, (Conv2dInterval, LinearInterval)):
                c.weight.data += sign * c.eps

    def validation_with_move_weights(self, dataloader):
        self.move_weights(-1)
        self.validation(dataloader, txt="Lower")
        self.restore_weights()

        self.move_weights(1)
        self.validation(dataloader, txt="Upper")
        self.restore_weights()

    def validation(self, dataloader, txt=""):
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

        self.log(' * {txt} Val Acc {acc.avg:.3f}, time {time:.2f}'.format(txt=txt, acc=acc, time=batch_timer.toc()))
        return acc.avg

    def _interval_based_bound(self, y0, idx, key):
        # requires last layer to be linear
        C = self.C[y0].t()
        cW = C @ (self.model.last[key].weight - self.model.last[key].eps)
        cb = C @ self.model.last[key].bias
        l, u = self.model.bounds
        return (cW.clamp(min=0) @ l[idx].t() + cW.clamp(max=0) @ u[idx].t() + cb[:, None]).t()

    def criterion(self, preds, targets, tasks, **kwargs):
        # The inputs and targets could come from single task or a mix of tasks
        # The network always makes the predictions with all its heads
        # The criterion will match the head and task to calculate the loss.
        loss, robust_loss, robust_err = 0, 0, 0
        if self.multihead:
            for t, t_preds in preds.items():
                inds = [i for i in range(len(tasks)) if tasks[i] == t]  # The index of inputs that matched specific task
                if len(inds) > 0:
                    t_preds = t_preds[inds]
                    t_target = targets[inds]
                    loss += self.criterion_fn(t_preds, t_target) * len(inds)

                    if self.eps_scheduler.current:
                        for y0 in range(len(self.C)):
                            if (t_target == y0).sum().item() > 0:
                                lower_bound = self._interval_based_bound(y0, t_target == y0, key=t)
                                # robust_loss += self.criterion_fn(-lower_bound, t_target[t_target == y0])
                                robust_loss += nn.CrossEntropyLoss(reduction='sum')(-lower_bound,
                                                                                    t_target[t_target == y0]) / t_target.size(0)

                                # increment when true label is not winning
                                robust_err += (lower_bound.min(dim=1)[0] < 0).sum().item()
                        robust_err /= len(t_target)

            loss /= len(targets)  # Average the total loss by the mini-batch size
            if self.eps_scheduler.current:
                loss *= self.kappa_scheduler.current
                loss += (1 - self.kappa_scheduler.current) * robust_loss

        else:
            pred = preds['All']
            # (Not 'ALL') Mask out the outputs of unseen classes for incremental class scenario
            if isinstance(self.valid_out_dim, int):
                pred = preds['All'][:, :self.valid_out_dim]
            loss = self.criterion_fn(pred, targets)
            if self.eps_scheduler.current:
                robust_loss, robust_err = 0, 0
                for y0 in range(len(self.C)):
                    if (targets == y0).sum().item() > 0:
                        lower_bound = self._interval_based_bound(y0, targets == y0, key="All")
                        # (Not 'ALL') Mask out the outputs of unseen classes for incremental class scenario
                        if isinstance(self.valid_out_dim, int):
                            lower_bound = lower_bound[:, :self.valid_out_dim]

                        robust_loss += self.criterion_fn(-lower_bound, targets[targets == y0])
                        # robust_loss += nn.CrossEntropyLoss(reduction='sum')(-lower_bound,
                        #                                                     targets[targets == y0]) / targets.size(0)

                        # increment when true label is not winning
                        robust_err += (lower_bound.min(dim=1)[0] < 0).sum().item()

                loss *= self.kappa_scheduler.current
                loss += (1 - self.kappa_scheduler.current) * robust_loss
                robust_err /= len(targets)

        return loss, robust_err

    def save_params(self):
        self.prev_weight, self.prev_eps, i = {}, {}, 0
        for block in self.model.children():
            if isinstance(block, nn.Sequential):
                for layer in block.children():
                    if isinstance(layer, (Conv2dInterval, LinearInterval)):
                        self.prev_weight[i] = layer.weight.data.detach().clone()
                        self.prev_eps[i] = layer.eps.clone()
                        i += 1

            elif isinstance(block, nn.ModuleDict) and not self.multihead:
                self.prev_weight[i] = block["All"].weight.data.detach().clone()
                self.prev_eps[i] = block["All"].eps.clone()
                i += 1

            elif isinstance(block, (Conv2dInterval, LinearInterval)):
                self.prev_weight[i] = block.weight.data.detach().clone()
                self.prev_eps[i] = block.eps.clone()
                i += 1

    def clip(self, data, low=None, upp=None):
        if low is not None:
            data = torch.where(data < low, low, data)
        if upp is not None:
            data = torch.where(data > upp, upp, data)
        return data

    def get_low_upp_eps(self, i, layer):
        eps = self.prev_eps[i]
        low_old = self.prev_weight[i] - eps
        upp_old = self.prev_weight[i] + eps
        # TODO: BETTER INTERVAL CLIPPING
        # low_new = layer.weight.data - layer.eps
        # upp_new = layer.weight.data + layer.eps
        #
        # low = torch.where(low_old < low_new, low_old, low_new)
        # upp = torch.where(upp_old > upp_new, upp_old, upp_new)
        # eps = self.clip(layer.eps, upp=self.prev_eps[i])
        # eps = torch.where(layer.eps < self.prev_eps[i], layer.eps, self.prev_eps[i])
        return low_old, upp_old, eps

    def clip_params(self):
        i = 0
        for c in self.model.children():
            if isinstance(c, nn.Sequential):
                for layer in c.children():
                    if isinstance(layer, (Conv2dInterval, LinearInterval)):
                        low, upp, eps = self.get_low_upp_eps(i, layer)
                        # layer.eps = eps
                        layer.weight.data = self.clip(layer.weight.data, low=low, upp=upp)
                        i += 1

            elif isinstance(c, nn.ModuleDict) and not self.multihead:
                low, upp, eps = self.get_low_upp_eps(i, c["All"])
                # c.eps = eps
                c["All"].weight.data = self.clip(c["All"].weight.data, low=low, upp=upp)
                i += 1

            elif isinstance(c, (Conv2dInterval, LinearInterval)):
                low, upp, eps = self.get_low_upp_eps(i, c)
                # c.eps = eps
                c.weight.data = self.clip(c.weight.data, low=low, upp=upp)
                i += 1

    def update_model(self, inputs, targets, tasks):
        out = self.forward(inputs)
        loss, robust_err = self.criterion(out, targets, tasks)
        self.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        nn.utils.clip_grad_norm_(self.model.parameters(), 1, norm_type=float('inf'))
        self.optimizer.step()
        if self.clipping and self.prev_eps:
            self.clip_params()

        # self.scheduler.step()
        self.kappa_scheduler.step()
        self.eps_scheduler.step()
        self.model.set_eps(self.eps_scheduler.current, trainable=self.config['eps_per_model'], head=self.current_head)

        return loss.detach(), robust_err, out

    def learn_batch(self, train_loader, val_loader=None):
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()

        schedule = self.schedule_stack.pop()
        for epoch in range(schedule):
            data_timer = Timer()
            batch_timer = Timer()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            acc = AverageMeter()
            robust_err = -1

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

                loss, robust_err, output = self.update_model(inputs, target, task)
                inputs = inputs.detach()
                target = target.detach()

                # measure accuracy and record loss
                acc = accumulate_acc(output, target, task, acc)
                losses.update(loss, inputs.size(0))

                batch_time.update(batch_timer.toc())  # measure elapsed time
                data_timer.toc()

            self.log(' * Train Acc {acc.avg:.3f}, Loss {loss.avg:.3f}'.format(loss=losses, acc=acc))
            self.log(f" * robust error: {robust_err}")

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
