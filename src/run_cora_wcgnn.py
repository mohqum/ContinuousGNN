import sys
import os
import copy
import json
import datetime

opt = dict()

opt['dataset'] = '../data/cora'
opt['hidden_dim'] = 40
opt['input_dropout'] = 0.5
opt['dropout'] = 0
opt['optimizer'] = 'rmsprop'
opt['lr'] = 0.00211
opt['decay'] = 5e-4
opt['self_link_weight'] = 0.947
opt['alpha']=0.6
opt['epoch'] = 400
opt['time']=14.3
opt['weight']=True
opt['decay_l1']=1e-4

def generate_command(opt):
    cmd = 'python train.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd

def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_))

for k in range(1):
    seed = k + 1
    opt['seed'] = 1
    run(opt)
    