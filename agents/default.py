from __future__ import print_function
import torch
import torch.nn as nn
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer

class NormalNN(nn.Module):
    '''
    Normal Neural Network with SGD for classification
    '''
    def __init__(self, agent_config):
        '''
        :param agent_config (dict): lr=float,momentum=float,weight_decay=float,
                                    schedule=[int],  # The last number in the list is the end of epoch
                                    model_type=str,model_name=str,out_dim={task:dim},model_weights=str
                                    force_single_head=bool
                                    print_freq=int
                                    gpuid=[int]
        '''
        super(NormalNN, self).__init__()
        self.log = print if agent_config['print_freq'] > 0 else lambda \
            *args: None  # Use a void function to replace the print
        self.config = agent_config
        # If out_dim is a dict, there is a list of tasks. The model will have a head for each task.
        self.multihead = True if len(self.config['out_dim'])>1 else False  # A convenience flag to indicate multi-head/task
        self.model = self.create_model()
        self.criterion_fn = nn.CrossEntropyLoss()
        if agent_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        self.init_optimizer()
        self.reset_optimizer = False
        self.valid_out_dim = 'ALL'  # Default: 'ALL' means all output nodes are active
                                    # Set a interger here for the incremental class scenario

    def init_optimizer(self):
        optimizer_arg = {'params':self.model.parameters(),
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'

        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['schedule'],
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
        for task,out_dim in cfg['out_dim'].items():
            model.last[task] = nn.Linear(n_feat,out_dim)

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
        for i, (input, target, task) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            output = self.predict(input)

            # Summarize the performance of all tasks, or 1 task, depends on dataloader.
            # Calculated by total number of data.
            acc = accumulate_acc(output, target, task, acc)

        self.train(orig_mode)

        self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
              .format(acc=acc,time=batch_timer.toc()))
        return acc.avg

    def criterion(self, preds, targets, tasks, **kwargs):
        # The inputs and targets could come from single task or a mix of tasks
        # The network always makes the predictions with all its heads
        # The criterion will match the head and task to calculate the loss.
        if self.multihead:
            loss = 0
            for t,t_preds in preds.items():
                inds = [i for i in range(len(tasks)) if tasks[i]==t]  # The index of inputs that matched specific task
                if len(inds)>0:
                    t_preds = t_preds[inds]
                    t_target = targets[inds]
                    loss += self.criterion_fn(t_preds, t_target) * len(inds)  # restore the loss from average
            loss /= len(targets)  # Average the total loss by the mini-batch size
        else:
            pred = preds['All']
            if isinstance(self.valid_out_dim, int):  # (Not 'ALL') Mask out the outputs of unseen classes for incremental class scenario
                pred = preds['All'][:,:self.valid_out_dim]
            loss = self.criterion_fn(pred, targets)
        return loss

    def update_model(self, inputs, targets, tasks):
        out = self.forward(inputs)
        loss = self.criterion(out, targets, tasks)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach(), out

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
            self.scheduler.step(epoch)
            for param_group in self.optimizer.param_groups:
                self.log('LR:',param_group['lr'])

            # Learning with mini-batch
            data_timer.tic()
            batch_timer.tic()
            self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
            for i, (input, target, task) in enumerate(train_loader):

                data_time.update(data_timer.toc())  # measure data loading time

                if self.gpu:
                    input = input.cuda()
                    target = target.cuda()

                loss, output = self.update_model(input, target, task)
                input = input.detach()
                target = target.detach()

                # measure accuracy and record loss
                acc = accumulate_acc(output, target, task, acc)
                losses.update(loss, input.size(0))

                batch_time.update(batch_timer.toc())  # measure elapsed time
                data_timer.toc()

                if ((self.config['print_freq']>0) and (i % self.config['print_freq'] == 0)) or (i+1)==len(train_loader):
                    self.log('[{0}/{1}]\t'
                          '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                          '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                          '{loss.val:.3f} ({loss.avg:.3f})\t'
                          '{acc.val:.2f} ({acc.avg:.2f})'.format(
                        i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, acc=acc))

            self.log(' * Train Acc {acc.avg:.3f}'.format(acc=acc))

            # Evaluate the performance of current task
            if val_loader != None:
                self.validation(val_loader)

    def learn_stream(self, data, label):
        assert False,'No implementation yet'

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
        if isinstance(self.model,torch.nn.DataParallel):
            # Get rid of 'module' before the name of states
            model_state = self.model.module.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        print('=> Saving model to:', filename)
        torch.save(model_state, filename + '.pth')
        print('=> Save Done')

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

def accumulate_acc(output, target, task, meter):
    if 'All' in output.keys(): # Single-headed model
        meter.update(accuracy(output['All'], target), len(target))
    else:  # outputs from multi-headed (multi-task) model
        for t, t_out in output.items():
            inds = [i for i in range(len(task)) if task[i] == t]  # The index of inputs that matched specific task
            if len(inds) > 0:
                t_out = t_out[inds]
                t_target = target[inds]
                meter.update(accuracy(t_out, t_target), len(inds))

    return meter
