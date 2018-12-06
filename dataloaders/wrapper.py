import torch
import torch.utils.data as data


class CacheClassLabel(data.Dataset):
    """
    A dataset wrapper that has a quick access to all labels of data.
    """
    def __init__(self, dataset, number_classes, labels):
        super(CacheClassLabel, self).__init__()
        self.dataset = dataset
        self.number_classes = number_classes
        self.labels = torch.LongTensor(labels)  # It will be used as tensor in loader generators.

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img,target = self.dataset[index]
        return img, target


class AppendName(data.Dataset):
    """
    A dataset wrapper that also return the name of the dataset/task
    """
    def __init__(self, dataset, name, first_class_ind=0):
        super(AppendName,self).__init__()
        self.dataset = dataset
        self.name = name
        self.first_class_ind = first_class_ind  # For remapping the class index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img,target = self.dataset[index]
        target = target + self.first_class_ind
        return img, target, self.name


class Subclass(data.Dataset):
    """
    A dataset wrapper that return the task name and remove the offset of labels (Let the labels start from 0)
    """
    def __init__(self, dataset, class_list, remap=True):
        '''
        :param dataset: (CacheClassLabel)
        :param class_list: (list) A list of integers
        :param remap: (bool) Ex: remap class [2,4,6 ...] to [0,1,2 ...]
        '''
        super(Subclass,self).__init__()
        assert isinstance(dataset, CacheClassLabel), 'dataset must be wrapped by CacheClassLabel'
        self.dataset = dataset
        self.class_list = class_list
        self.remap = remap
        self.indices = []
        for c in class_list:
            self.indices.extend((dataset.labels==c).nonzero().flatten().tolist())
        if remap:
            self.class_mapping = {c: i for i, c in enumerate(class_list)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img,target = self.dataset[self.indices[index]]
        if self.remap:
            raw_target = target.item() if isinstance(target,torch.Tensor) else target
            target = self.class_mapping[raw_target]
        return img, target


class Permutation(data.Dataset):
    """
    A dataset wrapper that permute the position of features
    """
    def __init__(self, dataset, permute_idx):
        super(Permutation,self).__init__()
        self.dataset = dataset
        self.permute_idx = permute_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img,target = self.dataset[index]
        shape = img.size()
        img = img.view(-1)[self.permute_idx].view(shape)
        return img, target


class Storage(data.Dataset):
    """
    A dataset wrapper used as a memory to store the data
    """
    def __init__(self):
        super(Storage, self).__init__()
        self.storage = []

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, index):
        return self.storage[index]

    def append(self,x):
        self.storage.append(x)

    def extend(self,x):
        self.storage.extend(x)
