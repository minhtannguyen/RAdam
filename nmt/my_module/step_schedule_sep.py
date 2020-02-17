# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler

@register_lr_scheduler('stepschedulesep')
class StepScheduleSep(FairseqLRScheduler):
    """Decay the LR on a fixed schedule."""

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with inverse_sqrt.'
                ' Consider --lr-scheduler=fixed instead.'
            )

        # then, decay prop. to the inverse square root of the update number
        # self.warmup_end_lr = warmup_end_lr * args.warmup_updates**0.5
        self.min_lr = args.min_lr

        # initial learning rate
        self.lr = args.lr[0]
        self.optimizer.set_lr(self.lr)

        self.max_update = args.max_update
        
        self.schedule_index = 0
        self.restart_index = 0

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--lr-shrink', default=0.1, type=float, metavar='LS',
                            help='shrink factor for annealing, lr_new = (lr * lr_shrink)')
        parser.add_argument('--schedule', type=int, nargs='+', default=[150000,],
                        help='Decrease learning rate at these epochs.')
        parser.add_argument('--restart-schedule', type=int, nargs='+', default=[350,],
                        help='Change restarting frequency to these values.')
        parser.add_argument('--restart-epoch', type=int, nargs='+', default=[150000,],
                        help='Change restart frequency at these epochs.')
        # fmt: on

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.args.schedule is None:
            self.lr  = self.args.lr[0]
        else:
            if num_updates in self.args.restart_epoch:
                if 'restarting_iter' in self.optimizer._optimizer.param_groups[0]:
                    for param_group in self.optimizer._optimizer.param_groups:
                        param_group['restarting_iter'] = self.args.restart_schedule[self.restart_index]
                self.restart_index += 1
            if num_updates in self.args.schedule:
                self.schedule_index += 1
                
            self.lr  = self.args.lr[0] * (self.args.lr_shrink ** self.schedule_index)
                
        self.optimizer.set_lr(self.lr)
        
#         print('\niter = %i\n'%num_updates)
#         print('\nrestarting iter = %i\n'%self.optimizer._optimizer.param_groups[0]['restarting_iter'])
#         print('\nlearning rate = %f\n'%self.optimizer._optimizer.param_groups[0]['lr'])
#         time.sleep(2)
        return self.lr
