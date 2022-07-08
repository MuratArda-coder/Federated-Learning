import torch
import numpy as np
import random

def set_seed(input):
    torch.manual_seed(input)
    np.random.seed(input)
    random.seed(input)
    print("In distributed training seed is : ", input)

def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()


def add(target, source):
    for name in target:
        target[name].data += source[name].data.clone()


def scale(target, scaling):
    for name in target:
        target[name].data = scaling * target[name].data.clone()


def subtract(target, source):
    for name in target:
        target[name].data -= source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


def average(target, sources):
    for name in target:
        target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()


def weighted_average(target, sources, weights):
    for name in target:
        summ = torch.sum(weights)
        n = len(sources)
        modify = [weight / summ * n for weight in weights]
        target[name].data = torch.mean(torch.stack([m * source[name].data for source, m in zip(sources, modify)]), dim=0).clone()


def majority_vote(target, sources, lr):
    for name in target:
        threshs = torch.stack([torch.max(source[name].data) for source in sources])
        mask = torch.stack([source[name].data.sign() for source in sources]).sum(dim=0).sign()
        target[name].data = (lr * mask).clone()

def count_bits(T):
  k = T.numel()
  B_pos = 0
  b_total = (B_pos+32)*k

  return b_total