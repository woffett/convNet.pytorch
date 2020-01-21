import argparse
import models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import os
from data import DataRegime, SampledDataRegime
from datetime import datetime
from utils.log import setup_logging, save_checkpoint, export_args_namespace
from utils.misc import torch_dtypes

def fetch_teacher_outputs(teacher_model, loader, teacher_path, dataset,
                          train=True, device='cpu'):
    '''
    Perform a single forward pass of the whole dataset to obtain
    the teacher model's logits.
    
    Inspired by: 
       https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/train.py
    '''
    
    teacher_dir = os.path.dirname(teacher_path)
    train_str = 'train' if train else 'val'
    output_path = os.path.join(teacher_dir, '%s_%s_outputs.pt' % (dataset,train_str))
    if os.path.exists(output_path):
        logging.info('Found saved teacher %s outputs!' % train_str)
        outputs = torch.load(output_path)
        return outputs

    logging.info('Generating teacher %s outputs...' % train_str)
    cuda = 'cuda' in device and torch.cuda.is_available()
    teacher_model.eval()
    if cuda:
        teacher_model = teacher_model.to(device)
    outputs = []
    for i, (inps, _) in enumerate(loader):
        if cuda:
            inps = inps.to(device)
        outputs.append(teacher_model(inps).data.cpu().numpy())

    torch.save(outputs, output_path)
    logging.info('Done!')

    return outputs

def construct_kd_loss(args):
    '''
    implementation from
    https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py

    Note: teacher outputs are pre-softmax log probabilities
    '''

    dtype = torch_dtypes.get(args.dtype)

    KLDiv = nn.KLDivLoss().to(args.device, dtype)
    CE = nn.CrossEntropyLoss().to(args.device, dtype)
    MSE = nn.MSELoss().to(args.device, dtype)
    alpha = args.alpha
    T = args.temperature

    def caruana_loss(outputs, labels, teacher_outputs):
        mse = MSE(outputs, teacher_outputs)
        ce = CE(outputs, labels)
        return (mse * alpha) + (ce * (1.0 - alpha))

    def hinton_loss(outputs, labels, teacher_outputs):
        kl = KLDiv(F.log_softmax(outputs / T, dim=1),
                   F.softmax(teacher_outputs / T, dim=1))
        ce = CE(outputs, labels)
        return (kl * alpha * T * T) + (ce * (1.0 - alpha))

    losses = {'hinton': hinton_loss, 'caruana': caruana_loss}

    return losses[args.distill_loss]
