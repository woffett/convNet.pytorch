import argparse
import models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
from data import DataRegime, SampledDataRegime
from tqdm import tqdm
from utils import setup_logging
from utils.meters import AverageMeter, accuracy

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='Training a student model via knowledge distillation through a teacher model')

parser.add_argument('--teacher', metavar='TEACHER', default='resnet',
                    choices=model_names,
                    help='teacher model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet)')
parser.add_argument('--teacher-path', type=str, required=True,
                    help='path to model weights')
parser.add_argument('--teacher-model-config', default='',
                    help='additional architecture configuration for teacher model')
parser.add_argument('--student', metavar='STUDENT', default='resnet',
                    choices=model_names,
                    help='student model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet)')
parser.add_argument('--student-model-config', default='',
                    help='additional architecture configuration for student model')
parser.add_argument('--input-size', type=int, default=None,
                    help='image input size')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--dataset', metavar='DATASET', default='cifar100',
                    help='dataset name or folder')
parser.add_argument('--datasets-dir', metavar='DATASETS_DIR', default='~/Datasets',
                    help='datasets dir')
parser.add_argument('--optimizer', metavar='OPT', type=str, default='SGD',
                    help='optimizer function')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='optimizer momentum')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='weight decay (default = 0)')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--loss', type=str, default='hinton',
                    choices=['hinton', 'caruana'],
                    help='the kind of loss function')
parser.add_argument('--temperature', default=6.0 , type=float,
                    help='Temperature for KD loss calculation')
parser.add_argument('--alpha', default=0.95, type=float,
                    help='Mixing hyperparam for KD loss calculation')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--device', default='cuda',
                    help='device assignment ("cpu" or "cuda")')
parser.add_argument('--print-freq', default=10, type=int,
                    help='how often to print out logs')

def fetch_teacher_outputs(teacher, loader, device='cpu'):
    '''
    Perform a signle forward pass of the whole dataset to obtain
    the teacher model's logits.
    
    Inspired by: 
       https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/train.py
    '''
    cuda = 'cuda' in device and torch.cuda.is_available()
    teacher_model.eval()
    if cuda:
        teacher_model = teacher_model.to(device)
    outputs = []
    for i, (inps, _) in enumerate(loader):
        if cuda:
            inps = inps.to(device)
        outputs.append(teacher_model(inps).data.cpu().numpy())

    return outputs

def construct_kd_loss(args):
    '''
    implementation from
    https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py

    Note: teacher outputs are pre-softmax log probabilities
    '''

    KLDiv = nn.KLDivLoss()
    CE = nn.CrossEntropyLoss()
    MSE = nn.MSELoss()
    T = args.temperature
    alpha = args.alpha

    def caruana_loss(outputs, labels, teacher_outputs):
        return MSE(outputs, teacher_outputs)

    def hinton_loss(outputs, labels, teacher_outputs):
        kl = KLDiv(F.log_softmax(outputs / T, dim=1),
                   F.softmax(teacher_outputs / T, dim=1))
        ce = CE(outputs, labels)
        return (kl * alpha * T * T) + (ce * (1.0 - alpha))

    losses = {'hinton': hinton_loss, 'caruana': caruana_loss}

    return lossses[args.loss]
    
def train(model, teacher_outputs, loader, optimizer, criterion,
          print_freq=10, device='cpu'):

    # setup meters
    meters = {name: AverageMeter()
              for name in ['step', 'data', 'loss', 'prec1', 'prec5']}
    end = time.time()
    
    # setup model for training
    model.train()
    cuda = ('cuda' in device) and torch.cuda.is_available()
    if cuda:
        model = model.to(device)
        
    # main training loop
    for i, (inps, targets) in tqdm(enumerate(loader)):
        # setup
        optimizer.zero_grad()
        if cuda:
            inps = inps.to(device)
            targets = targets.to(device)

        # forward
        outputs = model(inps)
        # backward
        teacher_outputs = torch.from_numpy(teacher_outputs[i])
        if cuda:
            teacher_outputs = teacher_outputs.to(device)
        loss = criterion(outputs, labels, teacher_outputs)
        loss.backward()
        optimizer.step()

        # measure accuracy and losses
        prec1, prec5 = accuracy(output, target, topk=(1,5))
        meters['loss'].update(float(loss), inputs.size(0))
        meters['prec1'].update(float(prec1), inputs.size(0))
        meters['prec5'].update(float(prec5), inputs.size(0))
        meters['step'].update(time.time() - end)
        end = time.time()

        # logging
        if (i % print_freq == 0) or (i == len(loader) - 1):
            report = str('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {meters[step].val:.3f} ({meters[step].avg:.3f})\t'
                         'Data {meters[data].val:.3f} ({meters[data].avg:.3f})\t'
                         'Loss {meters[loss].val:.4f} ({meters[loss].avg:.4f})\t'
                         'Prec@1 {meters[prec1].val:.3f} ({meters[prec1].avg:.3f})\t'
                         'Prec@5 {meters[prec5].val:.3f} ({meters[prec5].avg:.3f})\t'
                         .format(
                             self.epoch, i, len(data_loader),
                             phase='TRAINING',
                             meters=meters))
            logging.info(report)
        

def main(args):
    # first, set the random seed
    logging.info('*** Setting seed to %d ***' % args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    
    # load dataset
    logging.info('*** Loading %s dataset... ***' % args.dataset)
    data = DataRegime(getattr(model, 'data_eval_regime', None),
                      defaults={'datasets_path': args.datasets_dir,
                                'name': args.dataset,
                                'split': None,
                                'augment': False,
                                'input_size': args.input_size,
                                'batch_size': args.eval_batch_size,
                                'shuffle': False,
                                'num_workers': args.workers,
                                'pin_memory': True,
                                'drop_last': False})
    loader = data.get_loader()
    logging.info('Done!')

    # create teacher model
    logging.info('*** Loading teacher model... (%s) ***' % args.teacher)
    teacher = models.__dict__[args.teacher]
    teacher_config = {'dataset': args.dataset} \
        if args.teacher not in ['efficientnet', 'mobilenet'] \
           else dict()

    if args.teacher_model_config is not '':
        teacher_config = dict(model_config,
                            **literal_eval(args.teacher_model_config))
        
    teacher = teacher(**teacher_config)
    teacher_ckpt = torch.load(args.teacher_path)
    teacher.load_state_dict(teacher_ckpt['state_dict'])
    
    teacher_outputs = fetch_teacher_outputs(teacher, loader)

    # clear teacher from memory
    del teacher
    del teacher_ckpt

    logging.info('Done!')

    # create student model
    logging.info('*** Loading student model... (%s) ***' % args.student)
    model = models.__dict__[args.student]
    model_config = {'dataset': args.dataset} \
        if args.student not in ['efficientnet', 'mobilenet'] \
           else dict()
    if args.student_model_config is not '':
        model_config = dict(model_config,
                            **literal_eval(args.student_model_config))
        
    model = model(**model_config)

    logging.info('Done!')

    # create optimizer
    optim_args = {
        'epoch': 0,
        'optimizer': args.optimizer,
        'lr': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
    }
    optimizer = OptimRegime(model, optim_args)

    # construct knowledge distillation loss fn
    criterion = construct_kd_loss(args)

    # train using teacher outputs
    training_steps = 0
    for epoch in range(args.epochs):
        training_steps = epoch * len(train_data)
        optimizer.update(epoch, training_steps)
        train(model, teacher_outputs, loader, optimizer, criterion,
              print_freq=args.print_freq, device=args.device)

if __name__ == '__main__':
    args = parser.parse_rags()
    main(args)
