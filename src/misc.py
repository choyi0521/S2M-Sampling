import random
import numpy as np
import torch
from itertools import chain


class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class make_empty_object(object):
    pass
    

def flatten_dict(init_dict):
    res_dict = {}
    if type(init_dict) is not dict:
        return res_dict

    for k, v in init_dict.items():
        if type(v) == dict:
            res_dict.update(flatten_dict(v))
        else:
            res_dict[k] = v
    return res_dict


def setattr_cls_from_kwargs(cls, kwargs):
    kwargs = flatten_dict(kwargs)
    for key in kwargs.keys():
        value = kwargs[key]
        setattr(cls, key, value)


def dict2clsattr(train_configs, model_configs):
    cfgs = {}
    for k, v in chain(train_configs.items(), model_configs.items()):
        cfgs[k] = v

    class cfg_container: pass
    setattr_cls_from_kwargs(cfg_container, cfgs)
    return cfg_container


def fix_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


def load_checkpoint(model, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def set_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

def set_deterministic_op_trainable(m):
    if isinstance(m, torch.nn.modules.conv.Conv2d):
        m.train()
    if isinstance(m, torch.nn.modules.conv.ConvTranspose2d):
        m.train()
    if isinstance(m, torch.nn.modules.linear.Linear):
        m.train()
    if isinstance(m, torch.nn.modules.Embedding):
        m.train()
