import torch
from .default import NormalNN
from .regularization import SI, EWC, EWC_online
from .exp_replay import Naive_Rehearsal, GEM
from modules.criterions import BCEauto

def init_zero_weights(m):
    with torch.no_grad():
        if type(m) == torch.nn.Linear:
            m.weight.zero_()
            m.bias.zero_()
        elif type(m) == torch.nn.ModuleDict:
            for l in m.values():
                init_zero_weights(l)
        else:
            assert False, 'Only support linear layer'


def NormalNN_reset_optim(agent_config):
    agent = NormalNN(agent_config)
    agent.reset_optimizer = True
    return agent


def NormalNN_BCE(agent_config):
    agent = NormalNN(agent_config)
    agent.criterion_fn = BCEauto()
    return agent


def SI_BCE(agent_config):
    agent = SI(agent_config)
    agent.criterion_fn = BCEauto()
    return agent


def SI_splitMNIST_zero_init(agent_config):
    agent = SI(agent_config)
    agent.damping_factor = 1e-3
    agent.reset_optimizer = True
    agent.model.last.apply(init_zero_weights)
    return agent


def SI_splitMNIST_rand_init(agent_config):
    agent = SI(agent_config)
    agent.damping_factor = 1e-3
    agent.reset_optimizer = True
    return agent


def EWC_BCE(agent_config):
    agent = EWC(agent_config)
    agent.criterion_fn = BCEauto()
    return agent


def EWC_mnist(agent_config):
    agent = EWC(agent_config)
    agent.n_fisher_sample = 60000
    return agent


def EWC_online_mnist(agent_config):
    agent = EWC(agent_config)
    agent.n_fisher_sample = 60000
    agent.online_reg = True
    return agent


def EWC_online_empFI(agent_config):
    agent = EWC(agent_config)
    agent.empFI = True
    return agent


def EWC_zero_init(agent_config):
    agent = EWC(agent_config)
    agent.reset_optimizer = True
    agent.model.last.apply(init_zero_weights)
    return agent


def EWC_rand_init(agent_config):
    agent = EWC(agent_config)
    agent.reset_optimizer = True
    return agent


def EWC_reset_optim(agent_config):
    agent = EWC(agent_config)
    agent.reset_optimizer = True
    return agent


def EWC_online_reset_optim(agent_config):
    agent = EWC_online(agent_config)
    agent.reset_optimizer = True
    return agent


def Naive_Rehearsal_100(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 100
    return agent


def Naive_Rehearsal_200(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 200
    return agent


def Naive_Rehearsal_400(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 400
    return agent


def Naive_Rehearsal_1100(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 1100
    return agent


def Naive_Rehearsal_1400(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 1400
    return agent


def Naive_Rehearsal_4000(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 4000
    return agent


def Naive_Rehearsal_4400(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 4400
    return agent


def Naive_Rehearsal_5600(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 5600
    return agent


def Naive_Rehearsal_16000(agent_config):
    agent = Naive_Rehearsal(agent_config)
    agent.memory_size = 16000
    return agent


def GEM_100(agent_config):
    agent = GEM(agent_config)
    agent.memory_size = 100
    return agent


def GEM_200(agent_config):
    agent = GEM(agent_config)
    agent.memory_size = 200
    return agent


def GEM_400(agent_config):
    agent = GEM(agent_config)
    agent.memory_size = 400
    return agent


def GEM_1100(agent_config):
    agent = GEM(agent_config)
    agent.memory_size = 1100
    return agent


def GEM_4000(agent_config):
    agent = GEM(agent_config)
    agent.memory_size = 4000
    return agent


def GEM_4400(agent_config):
    agent = GEM(agent_config)
    agent.memory_size = 4400
    return agent


def GEM_16000(agent_config):
    agent = GEM(agent_config)
    agent.memory_size = 16000
    return agent