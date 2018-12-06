import torch
from .default import NormalNN
from .regularization import SI, L2, EWC, MAS
from dataloaders.wrapper import Storage


class Memory(Storage):
    def reduce(self, m):
        self.storage = self.storage[:m]


class Naive_Rehearsal(NormalNN):

    def __init__(self, agent_config):
        super(Naive_Rehearsal, self).__init__(agent_config)
        self.task_count = 0
        self.memory_size = 1000
        self.task_memory = {}

    def learn_batch(self, train_loader, val_loader=None):
        # 1.Combine training set
        dataset_list = []
        for storage in self.task_memory.values():
            dataset_list.append(storage)
        dataset_list *= max(len(train_loader.dataset)//self.memory_size,1)  # Let old data: new data = 1:1
        dataset_list.append(train_loader.dataset)
        dataset = torch.utils.data.ConcatDataset(dataset_list)
        new_train_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=train_loader.batch_size,
                                                       shuffle=True,
                                                       num_workers=train_loader.num_workers)

        # 2.Update model as normal
        super(Naive_Rehearsal, self).learn_batch(new_train_loader, val_loader)

        # 3.Randomly decide the images to stay in the memory
        self.task_count += 1
        # (a) Decide the number of samples for being saved
        num_sample_per_task = self.memory_size // self.task_count
        num_sample_per_task = min(len(train_loader.dataset),num_sample_per_task)
        # (b) Reduce current exemplar set to reserve the space for the new dataset
        for storage in self.task_memory.values():
            storage.reduce(num_sample_per_task)
        # (c) Randomly choose some samples from new task and save them to the memory
        self.task_memory[self.task_count] = Memory()  # Initialize the memory slot
        randind = torch.randperm(len(train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
        for ind in randind:  # save it to the memory
            self.task_memory[self.task_count].append(train_loader.dataset[ind])


class Naive_Rehearsal_SI(Naive_Rehearsal, SI):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_SI, self).__init__(agent_config)


class Naive_Rehearsal_L2(Naive_Rehearsal, L2):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_L2, self).__init__(agent_config)


class Naive_Rehearsal_EWC(Naive_Rehearsal, EWC):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_EWC, self).__init__(agent_config)
        self.online_reg = True  # Online EWC


class Naive_Rehearsal_MAS(Naive_Rehearsal, MAS):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_MAS, self).__init__(agent_config)
