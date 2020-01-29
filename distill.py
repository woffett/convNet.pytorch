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

def fetch_model_outputs(model, loader, device='cpu', return_activations=False):
    '''
    Perform a single forward pass of the whole dataset to obtain the model's logits.
    
    Inspired by: 
       https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/train.py
    '''
    logging.info('Generating model %s outputs')
    cuda = 'cuda' in device and torch.cuda.is_available()
    outputs = []
    activations = []
    for (inps, _) in loader:
        if cuda:
            inps = inps.to(device)
        if return_activations:
            curr_output, curr_activations = model.forward(inps, return_activations=return_activations)
            outputs.append(curr_output)
            activations.append(curr_activations)
        else:
            curr_output = model.forward(inps, return_activations=return_activations)
            outputs.append(curr_output)
    logging.info('Done!')
    return (torch.cat(outputs),torch.cat(activations)) if return_activations else torch.cat(outputs)

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
    beta = args.beta
    T = args.temperature

    def ce_loss(outputs, activations, teacher_outputs, teacher_U, P, eos_scale, labels):
        return CE(outputs, labels)

    def mse_loss(outputs, activations, teacher_outputs, teacher_U, P, eos_scale, labels):
        mse = MSE(outputs, teacher_outputs)
        ce = CE(outputs, labels)
        if alpha == 0.0:
            return ce
        return (mse * alpha) + (ce * (1.0 - alpha))

    def kldiv_loss(outputs, activations, teacher_outputs, teacher_U, P, eos_scale, labels):
        kl = KLDiv(F.log_softmax(outputs / T, dim=1),
                   F.softmax(teacher_outputs / T, dim=1))
        ce = CE(outputs, labels)
        if alpha == 0.0:
            return ce
        return (kl * alpha * T * T) + (ce * (1.0 - alpha))

    def eos_loss(outputs, activations, teacher_outputs, teacher_U, P, eos_scale, labels):
        mse = MSE(outputs, teacher_outputs)  if alpha > 0 else 0
        eos = MSE(activations @ P, teacher_U) * eos_scale if beta > 0 else 0
        ce = CE(outputs, labels) if (1 - alpha - beta) > 0 else 0
        return (mse * alpha) + (eos * beta) + (ce * (1.0 - alpha - beta))

    losses = {'kldiv': kldiv_loss, 'mse': mse_loss, 'eos': eos_loss, 'ce': ce_loss}

    if alpha == 0.0:
        return ce_loss
    
    if args.distill_loss == 'ce' and alpha > 0.0:
        raise Exception('Cannot specify non-zero alpha with CE loss!')

    if args.distill_loss == 'kldiv' and args.temperature is None:
        raise Exception('Must specify temperature for KLDiv loss!')

    return losses[args.distill_loss]
