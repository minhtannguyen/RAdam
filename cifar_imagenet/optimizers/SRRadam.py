# -*- coding: utf-8 -*-
"""
Scheduled restarting rectified Adam
"""
import math
import torch
from torch.optim.optimizer import Optimizer, required

class SRRAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, iter_count=1, restarting_iter=50)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(SRRAdam, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(SRRAdam, self).__setstate__(state)
    
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

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            
            # Compute the momentum parameter
            momentum_weights = (group['iter_count'] - 1.)/(group['iter_count'] + 2.)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                #exp_avg.mul_(beta1).add_(1 - beta1, grad) # BW commented
                exp_avg.mul_(momentum_weights).add_(1 - momentum_weights, grad) # BW added

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        #step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step']) # BW commented
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - momentum_weights ** state['step']) # BW added
                    else:
                        #step_size = group['lr'] / (1 - beta1 ** state['step']) # BW commented
                        step_size = group['lr'] / (1 - momentum_weights ** state['step']) # BW added
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:                    
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


class SRAdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, use_variance=True, warmup = 4000, iter_count=1, restarting_iter=50):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, use_variance=True, warmup = warmup, iter_count=iter_count, restarting_iter=restarting_iter)
        
        print('======== Warmup: {} ========='.format(warmup))
        super(SRAdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SRAdamW, self).__setstate__(state)
    
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
#         global iter_idx
#         iter_idx += 1
        grad_list = list()
        mom_list = list()
        mom_2rd_list = list()

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            
            # Compute the momentum parameter
            momentum_weights = (group['iter_count'] - 1.)/(group['iter_count'] + 2.)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                #exp_avg.mul_(beta1).add_(1 - beta1, grad) # BW commented
                exp_avg.mul_(momentum_weights).add_(1 - momentum_weights, grad) # BW added

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                #bias_correction1 = 1 - beta1 ** state['step'] # BW commented
                bias_correction1 = 1 - momentum_weights ** state['step'] # BW added
                bias_correction2 = 1 - beta2 ** state['step']
                
                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-6 + state['step'] * (group['lr'] - 1e-6) / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss