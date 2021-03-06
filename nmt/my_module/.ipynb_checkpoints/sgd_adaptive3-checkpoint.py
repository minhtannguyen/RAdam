"""
SGD with adaptive Nesterov momentum, mu = (k)/(k+3) if k <=200; else k = 1
"""
import torch
from .optimizer import Optimizer, required
import torch.optim

import math
import types
import time

import torch


from fairseq.optim import FairseqOptimizer, register_optimizer

from tensorboardX import SummaryWriter
# writer = SummaryWriter(logdir='./log/ada/')
# # writer = SummaryWriter(logdir='./log/wmt/')

iter_idx = 0

@register_optimizer('srsgd')
class FairseqSRSGD(FairseqOptimizer):

    def __init__(self, args, params):
        super().__init__(args)

        self._optimizer = SGD_Adaptive(params, **self.optimizer_config)
        self._optimizer.name = args.tb_tag + '_' + self._optimizer.name

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--tb-tag', default="", type=str,
                            help='tb tag')
        parser.add_argument('--iter-count', type=int, default=1, metavar='IC',
                            help='iter count for SRAdam optimizer')
        parser.add_argument('--restarting-iter', type=int, default=20, metavar='RI',
                            help='restarting iter for SRAdam optimizer')
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
            'weight_decay': self.args.weight_decay,
            'iter_count': self.args.iter_count,
            'restarting_iter': self.args.restarting_iter,
        }

class SGD_Adaptive(torch.optim.Optimizer):
    """
    Stochastic gradient descent with Adaptively restarting (200 iters) Nesterov momentum.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate.
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        iter_count (integer): count the iterations mod 200
    Example:
         >>> optimizer = torch.optim.SGD_Adaptive(model.parameters(), lr=0.1, weight_decay=5e-4, iter_count=1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
        >>> iter_count = optimizer.update_iter()
    Formula:
        v_{t+1} = p_t - lr*g_t
        p_{t+1} = v_{t+1} + (iter_count)/(iter_count+3)*(v_{t+1} - v_t)
    """
    def __init__(self, params, lr=required, weight_decay=0., iter_count=1, restarting_iter=100):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if iter_count < 1:
            raise ValueError("Invalid iter count: {}".format(iter_count))
        if restarting_iter < 1:
            raise ValueError("Invalid iter total: {}".format(restarting_iter))
        
        defaults = dict(lr=lr, weight_decay=weight_decay, iter_count=iter_count, restarting_iter=restarting_iter)
        self.name = '{}_{}'.format(lr, weight_decay)
        super(SGD_Adaptive, self).__init__(params, defaults)
        
    @property
    def supports_memory_efficient_fp16(self):
        return True
    
    def __setstate__(self, state):
        super(SGD_Adaptive, self).__setstate__(state)
    
    def update_iter(self):
        idx = 1
        for group in self.param_groups:
            if idx == 1:
                '''
                group['iter_count'] += 1
                group['iter_total'] += 1
                if group['iter_count'] >= 40*(group['iter_total']/10000+1): #50: #100: #200 #TODO: add another total count, 20*(total_count%10000 + 1), we can reschedule this, "40 can be made smaller, larger factor*(group['iter_total']/10000)"
                    group['iter_count'] = 1
                '''
                group['iter_count'] += 1
                if group['iter_count'] >= group['restarting_iter']: #50: #100: #200 #TODO: add another total count, 20*(total_count%10000 + 1), we can reschedule this, "40 can be made smaller, larger factor*(group['iter_total']/10000)" 2**group['iter_total']
                    group['iter_count'] = 1
            idx += 1 
        #print('Iter11: ', group['iter_count'])
        return group['iter_count'], group['restarting_iter']
    
    def step(self, closure=None):
        """
        Perform a single optimization step.
        Arguments: closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = (group['iter_count'] - 1.)/(group['iter_count'] + 2.)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay !=0:
                    d_p.add_(weight_decay, p.data)
                
                param_state = self.state[p]
                
                if 'momentum_buffer' not in param_state:
                    buf0 = param_state['momentum_buffer'] = torch.clone(p.data).detach()
                else:
                    buf0 = param_state['momentum_buffer']
                
             # buf1 = p.data - momentum*group['lr']*d_p
                buf1 = p.data - group['lr']*d_p
                p.data = buf1 + momentum*(buf1 - buf0)
                param_state['momentum_buffer'] = buf1
        
        iter_count, iter_total = self.update_iter()
        
        return loss
