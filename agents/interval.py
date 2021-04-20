import torch
import torch.optim as opt
import torch.nn as nn
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
from interval.layers import LinearInterval, Conv2dInterval
from interval.hyperparam_scheduler import LinearScheduler
from torch.utils.tensorboard import SummaryWriter


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
        self.current_task = 1
        self.schedule_stack = []
        self.tb = SummaryWriter(log_dir=f"runs/{self.config['dataset_name']}_experiment/")
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
        self.scheduler = opt.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['milestones'],
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
        if len(self.prev_weight) == 0:
            self.save_params()
        moves = (0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
        for move in moves:
            self.move_weights(-move)
            self.validation(dataloader, txt=f"Lower {move}")
            self.restore_weights()

        for move in moves:
            self.move_weights(-move)
            self.validation(dataloader, txt=f"Upper {move}")
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
        # cb = C @ self.model.last[key].bias
        l, u = self.model.bounds
        # return (cW.clamp(min=0) @ l[idx].t() + cW.clamp(max=0) @ u[idx].t() + cb[:, None]).t()
        return (cW.clamp(min=0) @ l[idx].t() + cW.clamp(max=0) @ u[idx].t()).t()

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

                        # robust_loss += self.criterion_fn(-lower_bound, targets[targets == y0])
                        robust_loss += nn.CrossEntropyLoss(reduction='sum')(-lower_bound,
                                                                            targets[targets == y0]) / targets.size(0)

                        # increment when true label is not winning
                        robust_err += (lower_bound.min(dim=1)[0] < 0).sum().item()

                loss *= self.kappa_scheduler.current
                loss += (1 - self.kappa_scheduler.current) * robust_loss
                robust_err /= len(targets)

        return loss, robust_err, robust_loss

    def save_params(self):
        i = 0
        for block in self.model.children():
            if isinstance(block, nn.Sequential):
                for layer in block.children():
                    if isinstance(layer, (Conv2dInterval, LinearInterval)):
                        # eps = self.clip_intervals(i, layer)
                        # if eps is None:
                        #     self.prev_eps[i] = layer.eps.clone()
                        # else:
                        #     self.prev_eps[i] = eps
                        self.prev_weight[i] = layer.weight.data.detach().clone()
                        self.prev_eps[i] = layer.eps.clone()
                        i += 1

            elif isinstance(block, nn.ModuleDict) and not self.multihead:
                # eps = self.clip_intervals(i, block["All"])
                # if eps is None:
                #     self.prev_eps[i] = block["All"].eps.clone()
                # else:
                #     self.prev_eps[i] = eps
                self.prev_weight[i] = block["All"].weight.data.detach().clone()
                self.prev_eps[i] = block["All"].eps.clone()
                i += 1

            elif isinstance(block, (Conv2dInterval, LinearInterval)):
                # eps = self.clip_intervals(i, block)
                # if eps is None:
                #     self.prev_eps[i] = block.eps.clone()
                # else:
                #     self.prev_eps[i] = eps
                self.prev_weight[i] = block.weight.data.detach().clone()
                self.prev_eps[i] = block.eps.clone()
                i += 1

        self.tb.add_histogram("input/weight", self.model.input.weight, self.current_task)
        self.tb.add_histogram("input/eps", self.model.input.eps, self.current_task)
        self.tb.add_histogram("input/importance", self.model.input.importance, self.current_task)

        self.tb.add_histogram("c1/0/weight", self.model.c1[0].weight, self.current_task)
        self.tb.add_histogram("c1/0/eps", self.model.c1[0].eps, self.current_task)
        self.tb.add_histogram("c1/0/importance", self.model.c1[0].importance, self.current_task)

        self.tb.add_histogram("c1/2/weight", self.model.c1[2].weight, self.current_task)
        self.tb.add_histogram("c1/2/eps", self.model.c1[2].eps, self.current_task)
        self.tb.add_histogram("c1/2/importance", self.model.c1[2].importance, self.current_task)

        self.tb.add_histogram("c2/0/weight", self.model.c2[0].weight, self.current_task)
        self.tb.add_histogram("c2/0/eps", self.model.c2[0].eps, self.current_task)
        self.tb.add_histogram("c2/0/importance", self.model.c2[0].importance, self.current_task)

        self.tb.add_histogram("c2/2/weight", self.model.c2[2].weight, self.current_task)
        self.tb.add_histogram("c2/2/eps", self.model.c2[2].eps, self.current_task)
        self.tb.add_histogram("c2/2/importance", self.model.c2[2].importance, self.current_task)

        self.tb.add_histogram("c3/0/weight", self.model.c3[0].weight, self.current_task)
        self.tb.add_histogram("c3/0/eps", self.model.c3[0].eps, self.current_task)
        self.tb.add_histogram("c3/0/importance", self.model.c3[0].importance, self.current_task)

        self.tb.add_histogram("c3/2/weight", self.model.c3[2].weight, self.current_task)
        self.tb.add_histogram("c3/2/eps", self.model.c3[2].eps, self.current_task)
        self.tb.add_histogram("c3/2/importance", self.model.c3[2].importance, self.current_task)

        self.tb.add_histogram('fc1/weight', self.model.fc1[0].weight, self.current_task)
        self.tb.add_histogram("fc1/eps", self.model.fc1[0].eps, self.current_task)
        self.tb.add_histogram("fc1/importance", self.model.fc1[0].importance, self.current_task)


        # # self.tb.add_histogram('fc1/bias', self.model.fc1.bias, self.current_task)
        # self.tb.add_histogram('fc1/weight', self.model.fc1.weight, self.current_task)
        # self.tb.add_histogram("fc1/eps", self.model.fc1.eps, self.current_task)
        # self.tb.add_histogram("fc1/importance", self.model.fc1.importance, self.current_task)
        #
        # # self.tb.add_histogram('fc2/bias', self.model.fc2.bias, self.current_task)
        # self.tb.add_histogram('fc2/weight', self.model.fc2.weight, self.current_task)
        # self.tb.add_histogram("fc2/eps", self.model.fc2.eps, self.current_task)
        # self.tb.add_histogram("fc2/importance", self.model.fc2.importance, self.current_task)

        # self.tb.add_histogram('last/bias', self.model.last[self.current_head].weight, self.current_task)
        self.tb.add_histogram('last/weight', self.model.last[self.current_head].weight, self.current_task)
        self.tb.add_histogram("last/eps", self.model.last[self.current_head].eps, self.current_task)
        self.tb.add_histogram("last/importance", self.model.last[self.current_head].importance, self.current_task)

        self.tb.flush()

        # i = 0
        # for block in self.model.children():
        #     if isinstance(block, nn.Sequential):
        #         for layer in block.children():
        #             if isinstance(layer, (Conv2dInterval, LinearInterval)):
        #                 self.prev_weight[i] = layer.weight.data.detach().clone()
        #                 self.prev_eps[i] = layer.eps.clone()
        #                 # compute upper and lower bound for previous weights
        #                 prev_lower = self.prev_weight[i] - self.prev_eps[i]
        #                 prev_upper = self.prev_weight[i] + self.prev_eps[i]
        #                 # compute upper and lower bound for current weights
        #                 cur_lower = layer.weight.data - layer.eps
        #                 cur_upper = layer.weight.data + layer.eps
        #                 # compute the interval intersection as the new previous weights
        #                 new_lower = torch.max(prev_lower, cur_lower)
        #                 new_upper = torch.min(prev_upper, cur_upper)
        #                 assert (new_lower <= new_upper).all()
        #                 # transform to weight + eps form
        #                 new_prev_weight = (new_lower + new_upper) / 2
        #                 new_prev_eps = torch.abs(new_lower - new_upper) / 2
        #                 # save
        #                 self.prev_weight[i] = new_prev_weight.detach().clone()
        #                 self.prev_eps[i] = new_prev_eps.detach().clone()
        #                 i += 1
        #
        #     elif isinstance(block, nn.ModuleDict) and not self.multihead:
        #         for _, layer in block.items():
        #             self.prev_weight[i] = layer.weight.data.detach().clone()
        #             self.prev_eps[i] = layer.eps.clone()
        #             prev_lower = self.prev_weight[i] - self.prev_eps[i]
        #             prev_upper = self.prev_weight[i] + self.prev_eps[i]
        #             cur_lower = layer.weight.data - layer.eps
        #             cur_upper = layer.weight.data + layer.eps
        #             new_lower = torch.max(prev_lower, cur_lower)
        #             new_upper = torch.min(prev_upper, cur_upper)
        #             assert (new_lower <= new_upper).all()
        #             new_prev_weight = (new_lower + new_upper) / 2
        #             new_prev_eps = torch.abs(new_lower - new_upper) / 2
        #             self.prev_weight[i] = new_prev_weight.detach().clone()
        #             self.prev_eps[i] = new_prev_eps.detach().clone()
        #             i += 1
        #
        #     elif isinstance(block, (Conv2dInterval, LinearInterval)):
        #         self.prev_weight[i] = block.weight.data.detach().clone()
        #         self.prev_eps[i] = block.eps.clone()
        #         prev_lower = self.prev_weight[i] - self.prev_eps[i]
        #         prev_upper = self.prev_weight[i] + self.prev_eps[i]
        #         cur_lower = block.weight.data - block.eps
        #         cur_upper = block.weight.data + block.eps
        #         new_lower = torch.max(prev_lower, cur_lower)
        #         new_upper = torch.min(prev_upper, cur_upper)
        #         assert (new_lower <= new_upper).all()
        #         new_prev_weight = (new_lower + new_upper) / 2
        #         new_prev_eps = torch.abs(new_lower - new_upper) / 2
        #         self.prev_weight[i] = new_prev_weight.detach().clone()
        #         self.prev_eps[i] = new_prev_eps.detach().clone()
        #         i += 1

    def clip_weights(self, i, weights):
        low_old = self.prev_weight[i] - self.prev_eps[i]
        upp_old = self.prev_weight[i] + self.prev_eps[i]
        weights = torch.max(low_old, weights)
        weights = torch.min(upp_old, weights)
        return weights

    def clip_intervals(self, i, layer):
        # if self.current_task < 2:
        #     return None
        eps_old = self.prev_eps[i]

        assert (eps_old >= 0).all()
        low_old = self.prev_weight[i] - eps_old
        upp_old = self.prev_weight[i] + eps_old

        low_new = layer.weight.data - layer.eps
        upp_new = layer.weight.data + layer.eps

        assert (low_old <= layer.weight.data).all()
        assert (upp_old >= layer.weight.data).all()

        low = torch.max(low_old, low_new)
        upp = torch.min(upp_old, upp_new)
        assert (low <= upp).all(), print(low[low <= upp], upp[low <= upp])

        # low_pos, low_neg = low.clamp(min=0), low.clamp(max=0)
        # upp_pos, upp_neg = upp.clamp(min=0), upp.clamp(max=0)
        #
        # new_prev_weight_low = (low_pos + upp_neg)
        # new_prev_weight_upp = (low_neg + upp_pos)
        #
        # new_prev_weight = (new_prev_weight_low + new_prev_weight_upp) / 2
        # eps_new = torch.abs(new_prev_weight_upp - new_prev_weight_low) / 2

        new_prev_weight = (low + upp) / 2
        eps_new = torch.abs(low - upp) / 2
        # eps_new = torch.min(layer.eps, eps_old)
        # eps_new = torch.min(torch.abs(low - new_prev_weight), torch.abs(upp - new_prev_weight))

        # assert (eps_old >= eps_new).all()

        self.prev_weight.update(i=new_prev_weight.detach().clone())
        self.prev_eps.update(i=eps_new.clone())

        return eps_new

    def clip_params(self):
        i = 0
        for c in self.model.children():
            if isinstance(c, nn.Sequential):
                for layer in c.children():
                    if isinstance(layer, (Conv2dInterval, LinearInterval)):
                        layer.weight.data = self.clip_weights(i, layer.weight.data)
                        layer.eps = self.clip_intervals(i, layer)
                        # layer.eps = torch.min(layer.eps, self.prev_eps[i])
                        i += 1

            elif isinstance(c, nn.ModuleDict) and not self.multihead:
                c["All"].weight.data = self.clip_weights(i, c["All"].weight.data)
                c["All"].eps = self.clip_intervals(i, c["All"])
                # c["All"].eps = torch.min(c["All"].eps, self.prev_eps[i])
                i += 1

            elif isinstance(c, (Conv2dInterval, LinearInterval)):
                c.weight.data = self.clip_weights(i, c.weight.data)
                c.eps = self.clip_intervals(i, c)
                # c.eps = torch.min(c.eps, self.prev_eps[i])
                i += 1

    def update_model(self, inputs, targets, tasks):
        out = self.forward(inputs)
        loss, robust_err, robust_loss = self.criterion(out, targets, tasks)
        self.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        nn.utils.clip_grad_norm_(self.model.parameters(), 1, norm_type=float('inf'))
        self.optimizer.step()
        if self.clipping and self.prev_eps:
            self.clip_params()

        self.kappa_scheduler.step()
        self.eps_scheduler.step()

        self.model.set_eps(self.eps_scheduler.current, trainable=self.config['eps_per_model'], head=self.current_head)
        return loss.detach(), robust_err, robust_loss, out

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
            robust_err, robust_loss = -1, -1

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

                loss, robust_err, robust_loss, output = self.update_model(inputs, target, task)
                inputs = inputs.detach()
                target = target.detach()
                self.tb.add_scalar(f"Loss/train - task {self.current_task}", loss, epoch)
                self.tb.add_scalar(f"Robust error/train - task {self.current_task}", robust_err, epoch)

                # measure accuracy and record loss
                acc = accumulate_acc(output, target, task, acc)
                losses.update(loss, inputs.size(0))

                batch_time.update(batch_timer.toc())  # measure elapsed time
                data_timer.toc()

            self.log(' * Train Acc {acc.avg:.3f}, Loss {loss.avg:.3f}'.format(loss=losses, acc=acc))
            self.log(f" * , robust loss: {robust_loss:.3f} robust error: {robust_err:.8f}")

            # Evaluate the performance of current task
            if val_loader is not None:
                self.validation(val_loader)

            self.scheduler.step()
            self.tb.flush()

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
