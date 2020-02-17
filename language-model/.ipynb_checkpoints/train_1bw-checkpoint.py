from __future__ import print_function
import datetime
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
import pickle
import math
import random

import fcntl

from model_word_ada.LM import LM
from model_word_ada.basic import BasicRNN
from model_word_ada.ddnet import DDRNN
from model_word_ada.lsradam import LSRAdam
from model_word_ada.lsadam import LSAdam
from model_word_ada.ldnet import LDRNN
from model_word_ada.densenet import DenseRNN
from model_word_ada.dataset import LargeDataset, EvalDataset
from model_word_ada.adaptive import AdaptiveSoftmax
import model_word_ada.utils as utils

# additional optimizers
from optimizers.RAdam import *
from optimizers.AdamW import *
from optimizers.SRAdamW import *
from optimizers.SRRAdam import *

from logger import Logger

from tensorboardX import SummaryWriter

# from tensorboardX import SummaryWriter
# writer = SummaryWriter(logdir='./cps/gadam/log_1bw_full/')

import argparse
import json
import os
import sys
import itertools
import functools

parser = argparse.ArgumentParser(description='PyTorch OneBillionwords Training')
parser.add_argument('--dataset_folder', default='/data/billionwords/one_billion/')
parser.add_argument('--load_checkpoint', default='')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--sequence_length', type=int, default=20)
parser.add_argument('--hid_dim', type=int, default=2048)
parser.add_argument('--word_dim', type=int, default=300)
parser.add_argument('--label_dim', type=int, default=-1)
parser.add_argument('--layer_num', type=int, default=2)
parser.add_argument('--droprate', type=float, default=0.1)
parser.add_argument('--add_relu', action='store_true')
parser.add_argument('--layer_drop', type=float, default=0.5)
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--epoch', type=int, default=14)
parser.add_argument('--clip', type=float, default=5)
parser.add_argument('--update', choices=['Adam', 'Adagrad', 'Adadelta', 'SGD', 'LSRAdam', 'LSAdam', 'AdamW', 'RAdam', 'SRAdamW', 'SRRAdam'], default='Adam')
parser.add_argument('--rnn_layer', choices=['Basic', 'DDNet', 'DenseNet', 'LDNet'], default='Basic')
parser.add_argument('--rnn_unit', choices=['gru', 'lstm', 'rnn', 'bnlstm'], default='lstm')
parser.add_argument('--lr', type=float, default=-1)
parser.add_argument('--schedule', type=lambda t: [int(tup) for tup in t.split(',')], default=[9])
parser.add_argument('--cut_off', nargs='+', default=[4000,40000,200000])
parser.add_argument('--interval', type=int, default=100)
parser.add_argument('--check_interval', type=int, default=4000)
parser.add_argument('--checkpath', default='./cps/gadam/')
parser.add_argument('--model_name', default='adam')
parser.add_argument('--sigma', default=0.1, type=float, help='sigma in LS(R)Adam')
parser.add_argument('--beta1', default=0.9, type=float,
                    help='beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float,
                    help='beta2 for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--warmup', default=0, type=float,
                    help='warmup steps for adam')
parser.add_argument('--restart-schedule', type=int, nargs='+', default=[80, 200, 500, 1000],
                        help='Restart at after these amounts of epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--manualSeed', type=int, help='manual seed')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
    
# logger
if not os.path.exists(args.checkpath): os.makedirs(args.checkpath)
writer = SummaryWriter(os.path.join(args.checkpath, 'tensorboard')) # write to tensorboard

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
torch.cuda.set_device(use_cuda)

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_ppl = np.inf  # best test accuracy

def main():
    global best_ppl
    
    print('loading dataset')
    dataset = pickle.load(open(args.dataset_folder + 'test.pk', 'rb'))
    w_map, test_data, range_idx = dataset['w_map'], dataset['test_data'], dataset['range']

    cut_off = args.cut_off + [len(w_map) + 1]

    train_loader = LargeDataset(args.dataset_folder, range_idx, args.batch_size, args.sequence_length)
    test_loader = EvalDataset(test_data, args.batch_size)

    print('building model')

    rnn_map = {'Basic': BasicRNN, 'DDNet': DDRNN, 'DenseNet': DenseRNN, 'LDNet': functools.partial(LDRNN, layer_drop = args.layer_drop)}
    rnn_layer = rnn_map[args.rnn_layer](args.layer_num, args.rnn_unit, args.word_dim, args.hid_dim, args.droprate)

    if args.label_dim > 0:
        soft_max = AdaptiveSoftmax(args.label_dim, cut_off)
    else:
        soft_max = AdaptiveSoftmax(rnn_layer.output_dim, cut_off)

    lm_model = LM(rnn_layer, soft_max, len(w_map), args.word_dim, args.droprate, label_dim = args.label_dim, add_relu=args.add_relu)
    lm_model.rand_ini()
    # lm_model.cuda()
    
    # set up optimizers
    optim_map = {'Adam' : optim.Adam, 'Adagrad': optim.Adagrad, 'Adadelta': optim.Adadelta, 'SGD': functools.partial(optim.SGD, momentum=0.9), 'LSRAdam':LSRAdam, 'LSAdam': LSAdam, 'AdamW': AdamW, 'RAdam': RAdam, 'SRAdamW': SRAdamW, 'SRRAdam': SRRAdam}
    if args.update.lower() == 'lsradam' or args.update.lower == 'lsadam':
            optimizer = optim_map[args.update](lm_model.parameters(), lr=args.lr*((1.+4.*args.sigma)**(0.25)), 
                           betas=(args.beta1, args.beta2),
                           weight_decay=args.weight_decay, 
                           sigma=args.sigma) 
    elif args.update.lower() == 'radam':
        optimizer = optim_map[args.update](lm_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif args.update.lower() == 'adamw':
        optimizer = optim_map[args.update](lm_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay, warmup=args.warmup)
    elif args.update.lower() == 'sradamw':
        iter_count = 1
        optimizer = optim_map[args.update](lm_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), iter_count=iter_count, weight_decay=args.weight_decay, warmup = args.warmup, restarting_iter=args.restart_schedule[0]) 
    elif args.update.lower() == 'srradam':
        #NOTE: need to double-check this
        iter_count = 1
        optimizer = optim_map[args.update](lm_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), iter_count=iter_count, weight_decay=args.weight_decay, warmup = args.warmup, restarting_iter=args.restart_schedule[0]) 
    else:
        if args.lr > 0:
            optimizer=optim_map[args.update](lm_model.parameters(), lr=args.lr)
        else:
            optimizer=optim_map[args.update](lm_model.parameters())
            
    # Resume
    title = 'onebillionword-' + args.rnn_layer
    logger = Logger(os.path.join(args.checkpath, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Train PPL', 'Valid PPL'])
    
    if args.load_checkpoint:
        if os.path.isfile(args.load_checkpoint):
            print("loading checkpoint: '{}'".format(args.load_checkpoint))
            checkpoint_file = torch.load(args.load_checkpoint, map_location=lambda storage, loc: storage)
            lm_model.load_state_dict(checkpoint_file['lm_model'], False)
            optimizer.load_state_dict(checkpoint_file['opt'], False)
        else:
            print("no checkpoint found at: '{}'".format(args.load_checkpoint))

    test_lm = nn.NLLLoss()
    
    test_lm.cuda()
    lm_model.cuda()
    
    batch_index = 0
    epoch_loss = 0
    full_epoch_loss = 0
    best_train_ppl = float('inf')
    cur_lr = args.lr
    
    schedule_index = 1

    try:
        for indexs in range(args.epoch):

            print('#' * 89)
            print('Start: {}'.format(indexs))

            if args.optimizer.lower() == 'sradamw':
                if indexs in args.schedule:
                    optimizer = SRAdamW(lm_model.parameters(), lr=args.lr * (args.gamma**schedule_index), betas=(args.beta1, args.beta2), iter_count=iter_count, weight_decay=args.weight_decay, warmup = 0, restarting_iter=args.restart_schedule[schedule_index])
                    schedule_index += 1

            elif args.optimizer.lower() == 'srradam':
                if indexs in args.schedule:
                    optimizer = SRRAdam(lm_model.parameters(), lr=args.lr * (args.gamma**schedule_index), betas=(args.beta1, args.beta2), iter_count=iter_count, weight_decay=args.weight_decay, warmup = 0, restarting_iter=args.restart_schedule[schedule_index])
                    schedule_index += 1
            
            else:
                adjust_learning_rate(optimizer, indexs)
                
            logger.file.write('\nEpoch: [%d | %d] LR: %f' % (indexs + 1, args.epoch, state['lr']))
            
            iterator = train_loader.get_tqdm()
            full_epoch_loss = 0

            lm_model.train()

            for word_t, label_t in iterator:

                if 1 == train_loader.cur_idx:
                    lm_model.init_hidden()

                label_t = label_t.view(-1)

                lm_model.zero_grad()
                loss = lm_model(word_t, label_t)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm(lm_model.parameters(), args.clip)
                optimizer.step()
                
                if args.optimizer.lower() == 'sradamw' or args.optimizer.lower() == 'srradam'
                    iter_count, iter_total = optimizer.update_iter()

                batch_index += 1 

                if 0 == batch_index % args.interval:
                    s_loss = utils.to_scalar(loss)
                    writer.add_scalars('loss_tracking/train_loss', {args.model_name:s_loss}, batch_index)
                
                epoch_loss += utils.to_scalar(loss)
                full_epoch_loss += utils.to_scalar(loss)
                if 0 == batch_index % args.check_interval:
                    epoch_ppl = math.exp(epoch_loss / args.check_interval)
                    writer.add_scalars('loss_tracking/train_ppl', {args.model_name: epoch_ppl}, batch_index)
                    print('epoch_ppl: {} lr: {} @ batch_index: {}'.format(epoch_ppl, cur_lr, batch_index))
                    logger.file.write('epoch_ppl: {} lr: {} @ batch_index: {}'.format(epoch_ppl, cur_lr, batch_index))
                    epoch_loss = 0
    
            test_ppl = evaluate(test_loader, lm_model, test_lm, -1)
        
            is_best = test_ppl < best_ppl
            best_ppl = min(test_ppl, best_ppl)

            writer.add_scalars('loss_tracking/test_ppl', {args.model_name: test_ppl}, indexs)
            print('test_ppl: {} @ index: {}'.format(test_ppl, indexs))
            logger.file.write('test_ppl: {} @ index: {}'.format(test_ppl, indexs))
            
            save_checkpoint({
                'epoch': epoch + 1,
                'schedule_index': schedule_index,
                'lm_model': lm_model.state_dict(),
                'ppl': test_ppl,
                'best_ppl': best_ppl,
                'opt':optimizer.state_dict(),
            }, is_best, indexs, checkpoint=args.checkpath)

    except KeyboardInterrupt:

        print('Exiting from training early')
        logger.file.write('Exiting from training early')
        test_ppl = evaluate(test_loader, lm_model, test_lm, -1)
        writer.add_scalars('loss_tracking/test_ppl', {args.model_name: test_ppl}, args.epoch)
        
        is_best=False
        save_checkpoint({
                'epoch': epoch + 1,
                'schedule_index': schedule_index,
                'lm_model': lm_model.state_dict(),
                'ppl': test_ppl,
                'best_ppl': best_ppl,
                'opt':optimizer.state_dict(),
            }, is_best, indexs, checkpoint=args.checkpath)
    
    print('Best PPL:%f'%best_ppl)
    
    logger.file.write('Best PPL:%f'%best_ppl)  
    logger.close()
    
    with open("./all_results.txt", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write("%s\n"%args.checkpath)
        f.write("best_ppl %f\n\n"%best_ppl)
        fcntl.flock(f, fcntl.LOCK_UN)

def evaluate(data_loader, lm_model, criterion, limited = 76800):
    print('evaluating')
    lm_model.eval()

    iterator = data_loader.get_tqdm()

    lm_model.init_hidden()
    total_loss = 0
    total_len = 0
    for word_t, label_t in iterator:
        label_t = label_t.view(-1)
        tmp_len = label_t.size(0)
        output = lm_model.log_prob(word_t)
        total_loss += tmp_len * utils.to_scalar(criterion(autograd.Variable(output), label_t))
        total_len += tmp_len

        if limited >=0 and total_len > limited:
            break

    ppl = math.exp(total_loss / total_len)
    print('PPL: ' + str(ppl))

    return ppl

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(state, is_best, epoch, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    next_epoch = epoch + 1
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
    if next_epoch in args.schedule:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_epoch_%i.pth.tar'%epoch))
        
def adjust_learning_rate(optimizer, indexs):
    global state
    if indexs in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
            
if __name__ == '__main__':
    main()
    # writer.close()