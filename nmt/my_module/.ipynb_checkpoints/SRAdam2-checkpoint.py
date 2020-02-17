# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import types

import torch
import torch.optim

from fairseq.optim import FairseqOptimizer, register_optimizer

from tensorboardX import SummaryWriter
# writer = SummaryWriter(logdir='./log/ada/')
# # writer = SummaryWriter(logdir='./log/wmt/')

iter_idx = 0

@register_optimizer('sradam2')
class FairseqSRAdam2(FairseqOptimizer):

    def __init__(self, args, params):
        super().__init__(args, params)

        self._optimizer = SRAdam2(params, **self.optimizer_config)
        self._optimizer.name = args.tb_tag + '_' + self._optimizer.name

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--adam-betas', default='(0.9, 0.999)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for Adam optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--tb-tag', default="", type=str,
                            help='tb tag')
        parser.add_argument('--amsgrad', action='store_true')
        parser.add_argument('--adam-freeze', default=5000, type=float)
        parser.add_argument('--adam-no-correction1', action='store_true')
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr[0],
            'betas': eval(self.args.adam_betas),
            'eps': self.args.adam_eps,
            'weight_decay': self.args.weight_decay,
            'amsgrad': self.args.amsgrad,
            'adam_freeze': self.args.adam_freeze,
            'adam_no_correction1': self.args.adam_no_correction1,
        }

class SRAdam2(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, adam_freeze=5000, adam_no_correction1=False, iter_count=1, restarting_iter=50):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, adam_freeze=adam_freeze, adam_no_correction1=adam_no_correction1, iter_count=iter_count, restarting_iter=restarting_iter)
        self.name = '{}_{}_{}'.format(lr, betas[0], betas[1])
        super(SRAdam2, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True
    
    def update_iter(self):
        idx = 1
        for group in self.param_groups:
            if idx == 1:
                group['iter_count'] += 1
                if group['iter_count'] >= group['restarting_iter']:
                    group['iter_count'] = 1
            idx += 1
        return group['iter_count'], group['restarting_iter']
    
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        global iter_idx
        iter_idx += 1
        grad_list = list()
        mom_list = list()
        mom_2rd_list = list()

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = (group['iter_count'] - 1.)/(group['iter_count'] + 2.)
            # if 'adam_1k' in self.name:
            #     writer_iter = iter_idx - group['adam_freeze']
            # else:
            #     writer_iter = iter_idx

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                    if amsgrad:
                        state['max_exp_avg_sq'] = state['max_exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                exp_avg_sq.mul_(beta2).addcmul_(1-beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                if group['adam_no_correction1']:
                    bias_correction1 = 1
                else:
                    #bias_correction1 = (1 - beta1 ** state['step'])
                    bias_correction1 = (1 - momentum ** state['step'])

                bias_correction2 = (1 - beta2 ** state['step'])**0.5
                step_size = group['lr'] * bias_correction2 / bias_correction1


                if 'adam_1k' not in self.name or state['step'] > group['adam_freeze']:
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(1., grad)
                    grad = grad.add(momentum, buf)
                    
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    exp_avg.mul_(momentum).add_(1 - momentum, grad)
                    p_data_fp32.addcdiv_(-step_size, grad, denom)
                    p.data.copy_(p_data_fp32)
        return loss
