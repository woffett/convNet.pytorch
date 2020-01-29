import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from ast import literal_eval
from torch.nn.utils import clip_grad_norm_
from utils.meters import AverageMeter, accuracy
from utils.mixup import MixUp, CutMix
from random import sample
from distill import fetch_model_outputs
import models
try:
    import tensorwatch
    _TENSORWATCH_AVAILABLE = True
except ImportError:
    _TENSORWATCH_AVAILABLE = False

# profiling tools
from pytorch_memlab import profile, MemReporter


def _flatten_duplicates(inputs, target, batch_first=True, expand_target=True):
    duplicates = inputs.size(1)
    if not batch_first:
        inputs = inputs.transpose(0, 1)
    inputs = inputs.flatten(0, 1)

    if expand_target:
        if batch_first:
            target = target.view(-1, 1).expand(-1, duplicates)
        else:
            target = target.view(1, -1).expand(duplicates, -1)
        target = target.flatten(0, 1)
    return inputs, target


def _average_duplicates(outputs, target, batch_first=True):
    """assumes target is not expanded (target.size(0) == batch_size) """
    batch_size = target.size(0)
    reduce_dim = 1 if batch_first else 0
    if batch_first:
        outputs = outputs.view(batch_size, -1, *outputs.shape[1:])
    else:
        outputs = outputs.view(-1, batch_size, *outputs.shape[1:])
    outputs = outputs.mean(dim=reduce_dim)
    return outputs


def _mixup(mixup_modules, alpha, batch_size):
    mixup_layer = None
    if len(mixup_modules) > 0:
        for m in mixup_modules:
            m.reset()
        mixup_layer = sample(mixup_modules, 1)[0]
        mixup_layer.sample(alpha, batch_size)
    return mixup_layer


class Trainer(object):

    def __init__(self, model, criterion, optimizer=None,
                 device_ids=[0], device=torch.cuda, dtype=torch.float,
                 distributed=False, local_rank=-1, adapt_grad_norm=None,
                 mixup=None, cutmix=None, loss_scale=1., grad_clip=-1,
                 print_freq=100, dataset=None,
                 teacher=None, teacher_path=None, teacher_model_config=''):
        self._model = model
        self.criterion = criterion
        self.epoch = 0
        self.training_steps = 0
        self.optimizer = optimizer
        self.device = device
        self.dtype = dtype
        self.distributed = distributed
        self.local_rank = local_rank
        self.print_freq = print_freq
        self.grad_clip = grad_clip
        self.mixup = mixup
        self.cutmix = cutmix
        self.grad_scale = None
        self.loss_scale = loss_scale
        self.adapt_grad_norm = adapt_grad_norm
        self.watcher = None
        self.streams = {}
        self.dataset = dataset

        if distributed:
            self.model = nn.parallel.DistributedDataParallel(model,
                                                             device_ids=device_ids,
                                                             output_device=device_ids[0])
        elif device_ids and len(device_ids) > 1:
            self.model = nn.DataParallel(model, device_ids)
        else:
            self.model = model

        self.teacher = teacher
        self.teacher_path = teacher_path
        self.teacher_model_config = teacher_model_config
        if self.teacher is not None:
            self.load_teacher_model()

    def load_teacher_model(self):
        teacher_model = models.__dict__[self.teacher]
        teacher_config = {'dataset': self.dataset} \
                         if self.teacher not in ['efficientnet', 'mobilenet'] \
                            else dict()
        teacher_config = dict(teacher_config,
                              **literal_eval(self.teacher_model_config))
        teacher_model = teacher_model(**teacher_config)
        teacher_ckpt = torch.load(self.teacher_path)
        teacher_model.load_state_dict(teacher_ckpt['state_dict'])
        cuda = 'cuda' in self.device and torch.cuda.is_available()
        teacher_model.eval()
        if cuda:
            teacher_model = teacher_model.to(self.device)
        self.teacher_model = teacher_model

    def _set_student_and_teacher_outputs(self, train_data_loader, val_data_loader):
        with torch.no_grad():
            self.teacher_train_outputs, self.teacher_train_activations = fetch_model_outputs(
                self.teacher_model, train_data_loader, device=self.device, return_activations=True)
            self.teacher_val_outputs, self.teacher_val_activations = fetch_model_outputs(
                self.teacher_model, val_data_loader, device=self.device, return_activations=True)
            self.student_train_outputs, self.student_train_activations = fetch_model_outputs(
                self.model, train_data_loader, device=self.device, return_activations=True)
            self.student_val_outputs, self.student_val_activations = fetch_model_outputs(
                self.model, val_data_loader, device=self.device, return_activations=True)
            train_svd = torch.svd(self.teacher_train_activations)
            val_svd = torch.svd(self.teacher_val_activations)
            self.teacher_train_U = train_svd.U
            self.teacher_val_U = val_svd.U
            # we use eos_scale to scale the EOS loss during training, so that the same learning rates that were
            # good for MSE loss are also good for EOS loss.
            self.eos_scale = torch.norm(self.teacher_train_activations) / torch.norm(self.teacher_train_U)

    def get_teacher_output(self, training=False, curr_index=0, chunk_size=0):
        full_outputs = self.teacher_train_outputs if training else self.teacher_val_outputs
        full_activations = self.teacher_train_activations if training else self.teacher_val_activations
        full_U = self.teacher_train_U if training else self.teacher_val_U
        curr_outputs = full_outputs[curr_index:curr_index+chunk_size,:]
        curr_activations = full_activations[curr_index:curr_index+chunk_size,:]
        curr_U = full_U[curr_index:curr_index+chunk_size,:]
        return curr_outputs, curr_activations, curr_U

    def update_student_outputs(self, output, activations, training=False, curr_index=0, chunk_size=0):
        full_outputs = self.student_train_outputs if training else self.student_val_outputs
        full_activations = self.student_train_activations if training else self.student_val_activations
        full_outputs[curr_index:curr_index+chunk_size,:] = output
        full_activations[curr_index:curr_index+chunk_size,:] = activations

    def update_P(self):
        with torch.no_grad():
            X = self.student_train_activations
            self.P = torch.pinverse(X.t() @ X) @ X.t() @ self.teacher_train_U

    def compute_eos(self, training=False):
        teacher_U = self.teacher_train_U if training else self.teacher_val_U
        student_activations = self.student_train_activations if training else self.student_val_activations
        student_svd = torch.svd(student_activations)
        student_U = student_svd.U
        return torch.norm(student_U.t() @ teacher_U).item()**2 / teacher_U.shape[1]

    def _grad_norm(self, inputs_batch, target_batch, chunk_batch=1):
        self.model.zero_grad()
        for inputs, target in zip(inputs_batch.chunk(chunk_batch, dim=0),
                                  target_batch.chunk(chunk_batch, dim=0)):
            target = target.to(self.device)
            inputs = inputs.to(self.device, dtype=self.dtype)

            # compute output
            output = self.model(inputs)
            loss = self.criterion(output, target)

            if chunk_batch > 1:
                loss = loss / chunk_batch

            loss.backward()   # accumulate gradient
        grad = clip_grad_norm_(self.model.parameters(), float('inf'))
        return grad

    def _step(self, inputs_batch, target_batch, training=False, average_output=False, chunk_batch=1, curr_index=0):
        outputs = []
        total_loss = 0

        if training:
            self.optimizer.zero_grad()
            self.optimizer.update(self.epoch, self.training_steps)

        for i, (inputs, target) in enumerate(zip(inputs_batch.chunk(chunk_batch, dim=0),
                                                 target_batch.chunk(chunk_batch, dim=0))):
            target = target.to(self.device)
            inputs = inputs.to(self.device, dtype=self.dtype)

            mixup = None
            if training:
                self.optimizer.pre_forward()
                if self.mixup is not None or self.cutmix is not None:
                    input_mixup = CutMix() if self.cutmix else MixUp()
                    mix_val = self.mixup or self.cutmix
                    mixup_modules = [input_mixup]  # input mixup
                    mixup_modules += [m for m in self.model.modules()
                                      if isinstance(m, MixUp)]
                    mixup = _mixup(mixup_modules, mix_val, inputs.size(0))
                    inputs = input_mixup(inputs)

            # compute output
            output,activations = self.model.forward(inputs, return_activations=True)

            if mixup is not None:
                target = mixup.mix_target(target, output.size(-1))

            if average_output:
                if isinstance(output, list) or isinstance(output, tuple):
                    output = [_average_duplicates(out, target) if out is not None else None
                              for out in output]
                else:
                    output = _average_duplicates(output, target)
                    
            if self.teacher is not None:
                chunk_size = inputs.shape[0]
                teacher_output,_,teacher_U = self.get_teacher_output(training=training, curr_index=curr_index, chunk_size=chunk_size)
                self.update_student_outputs(output, activations, training=training, curr_index=curr_index, chunk_size=chunk_size)
                if curr_index % 1000 < chunk_size and training:
                    self.update_P()
                loss = self.criterion(output, activations, teacher_output, teacher_U, self.P, self.eos_scale, target)
                curr_index += chunk_size
            else:
                loss = self.criterion(output, target)
            grad = None

            if chunk_batch > 1:
                loss = loss / chunk_batch

            if isinstance(output, list) or isinstance(output, tuple):
                output = output[0]

            outputs.append(output.detach())
            total_loss += float(loss)

            if training:
                if i == 0:
                    self.optimizer.pre_backward()
                if self.grad_scale is not None:
                    loss = loss * self.grad_scale
                if self.loss_scale is not None:
                    loss = loss * self.loss_scale
                loss.backward()   # accumulate gradient

        if training:  # post gradient accumulation
            if self.loss_scale is not None:
                for p in self.model.parameters():
                    if p.grad is None:
                        continue
                    p.grad.data.div_(self.loss_scale)

            if self.grad_clip > 0:
                grad = clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()  # SGD step
            self.training_steps += 1

        outputs = torch.cat(outputs, dim=0)
        return outputs, total_loss, grad

    def forward(self, data_loader, num_steps=None, training=False, average_output=False, chunk_batch=1):
        meters = {name: AverageMeter()
                  for name in ['step', 'data', 'loss', 'prec1', 'prec5']}
        if training and self.grad_clip > 0:
            meters['grad'] = AverageMeter()

        batch_first = True
        if training and isinstance(self.model, nn.DataParallel) or chunk_batch > 1:
            batch_first = False

        def meter_results(meters):
            results = {name: meter.avg for name, meter in meters.items()}
            results['error1'] = 100. - results['prec1']
            results['error5'] = 100. - results['prec5']
            return results

        curr_index = 0
        end = time.time()
        for i, (inputs, target) in enumerate(data_loader):
            duplicates = inputs.dim() > 4  # B x D x C x H x W
            if training and duplicates and self.adapt_grad_norm is not None \
                    and i % self.adapt_grad_norm == 0:
                grad_mean = 0
                num = inputs.size(1)
                for j in range(num):
                    grad_mean += float(self._grad_norm(inputs.select(1, j), target))
                grad_mean /= num
                grad_all = float(self._grad_norm(
                    *_flatten_duplicates(inputs, target, batch_first)))
                self.grad_scale = grad_mean / grad_all
                logging.info('New loss scale: %s', self.grad_scale)

            # measure data loading time
            meters['data'].update(time.time() - end)
            if duplicates:  # multiple versions for each sample (dim 1)
                inputs, target = _flatten_duplicates(inputs, target, batch_first,
                                                     expand_target=not average_output)

            output, loss, grad = self._step(inputs, target,
                                            training=training,
                                            average_output=average_output,
                                            chunk_batch=chunk_batch,
                                            curr_index=curr_index)
            curr_index += inputs.shape[0]

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            meters['loss'].update(float(loss), inputs.size(0))
            meters['prec1'].update(float(prec1), inputs.size(0))
            meters['prec5'].update(float(prec5), inputs.size(0))
            if grad is not None:
                meters['grad'].update(float(grad), inputs.size(0))

            # measure elapsed time
            meters['step'].update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0 or i == len(data_loader) - 1:
                report = str('{phase} - Epoch: [{0}][{1}/{2}]\t'
                             'Time {meters[step].val:.3f} ({meters[step].avg:.3f})\t'
                             'Data {meters[data].val:.3f} ({meters[data].avg:.3f})\t'
                             'Loss {meters[loss].val:.4f} ({meters[loss].avg:.4f})\t'
                             'Prec@1 {meters[prec1].val:.3f} ({meters[prec1].avg:.3f})\t'
                             'Prec@5 {meters[prec5].val:.3f} ({meters[prec5].avg:.3f})\t'
                             .format(
                                 self.epoch, i, len(data_loader),
                                 phase='TRAINING' if training else 'EVALUATING',
                                 meters=meters))
                if 'grad' in meters.keys():
                    report += 'Grad {meters[grad].val:.3f} ({meters[grad].avg:.3f})'\
                        .format(meters=meters)
                logging.info(report)
                self.observe(trainer=self,
                             model=self._model,
                             optimizer=self.optimizer,
                             data=(inputs, target))
                self.stream_meters(meters,
                                   prefix='train' if training else 'eval')
                if training:
                    self.write_stream('lr',
                                      (self.training_steps, self.optimizer.get_lr()[0]))

            if num_steps is not None and i >= num_steps:
                break

        results = meter_results(meters)
        # Add EOS result
        results['eos'] = self.compute_eos(training=training)
        return results

    def train(self, data_loader, average_output=False, chunk_batch=1):
        # switch to train mode
        self.model.train()
        self.write_stream('epoch', (self.training_steps, self.epoch))
        return self.forward(data_loader, training=True, average_output=average_output, chunk_batch=chunk_batch)

    def validate(self, data_loader, average_output=False):
        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            return self.forward(data_loader, average_output=average_output, training=False)

    def calibrate_bn(self, data_loader, num_steps=None):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = None
                m.track_running_stats = True
                m.reset_running_stats()
        self.model.train()
        with torch.no_grad():
            return self.forward(data_loader, num_steps=num_steps, training=False)

    ###### tensorwatch methods to enable training-time logging ######

    def set_watcher(self, filename, port=0):
        if not _TENSORWATCH_AVAILABLE:
            return False
        if self.distributed and self.local_rank > 0:
            return False
        self.watcher = tensorwatch.Watcher(filename=filename, port=port)
        # default streams
        self._default_streams()
        self.watcher.make_notebook()
        return True

    def get_stream(self, name, **kwargs):
        if self.watcher is None:
            return None
        if name not in self.streams.keys():
            self.streams[name] = self.watcher.create_stream(name=name,
                                                            **kwargs)
        return self.streams[name]

    def write_stream(self, name, values):
        stream = self.get_stream(name)
        if stream is not None:
            stream.write(values)

    def stream_meters(self, meters_dict, prefix=None):
        if self.watcher is None:
            return False
        for name, value in meters_dict.items():
            if prefix is not None:
                name = '_'.join([prefix, name])
            value = value.val
            stream = self.get_stream(name)
            if stream is None:
                continue
            stream.write((self.training_steps, value))
        return True

    def observe(self, **kwargs):
        if self.watcher is None:
            return False
        self.watcher.observe(**kwargs)
        return True

    def _default_streams(self):
        self.get_stream('train_loss')
        self.get_stream('eval_loss')
        self.get_stream('train_prec1')
        self.get_stream('eval_prec1')
        self.get_stream('lr')
