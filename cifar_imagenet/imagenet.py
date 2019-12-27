'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import fcntl

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.radam import RAdam, AdamW
from utils.lsradam import LSRAdam, LSAdamW

from optimizers.sgd_adaptive3 import *
from optimizers.SRNadam import *
from optimizers.SRRadam import *

from tensorboardX import SummaryWriter

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer sgd|adam|radam')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--beta1', default=0.9, type=float,
                    help='beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float,
                    help='beta2 for adam')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--restart-schedule', type=int, nargs='+', default=[80, 200, 500, 1000],
                        help='Restart at after these amounts of epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--model_name', default='sgd')

# DALI
parser.add_argument('--dali_cpu', action='store_true',
                    help='Runs CPU based version of DALI pipeline.')

parser.add_argument("--local_rank", default=0, type=int)

# LSAdam
parser.add_argument('--sigma', default=0.1, type=float, help='sigma in LSAdam')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# logger
if not os.path.exists(args.checkpoint): os.makedirs(args.checkpoint)
writer = SummaryWriter(os.path.join(args.checkpoint, 'tensorboard')) # write to tensorboard

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=True)
        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        #logger.info('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


best_top1 = 0  # best test top1 accuracy
best_top5 = 0  # best test top5 accuracy
batch_time_global = AverageMeter()
data_time_global = AverageMeter()

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def main():
    global best_top1, best_top5
    
    args.world_size = 1
    
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data loading code    
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    crop_size = 224
    val_size = 256

    pipe = HybridTrainPipe(batch_size=args.train_batch, num_threads=args.workers, device_id=args.local_rank, data_dir=traindir, crop=crop_size, dali_cpu=args.dali_cpu)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    pipe = HybridValPipe(batch_size=args.test_batch, num_threads=args.workers, device_id=args.local_rank, data_dir=valdir, crop=crop_size, size=val_size)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    elif args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    baseWidth=args.base_width,
                    cardinality=args.cardinality,
                )
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adamw':
        optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay, warmup = 0)
    elif args.optimizer.lower() == 'radam':
        optimizer = RAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'lsadam': 
        optimizer = LSAdamW(model.parameters(), lr=args.lr*((1.+4.*args.sigma)**(0.25)), 
                           betas=(args.beta1, args.beta2),
                           weight_decay=args.weight_decay, 
                           sigma=args.sigma)
    elif args.optimizer.lower() == 'lsradam':
        sigma = 0.1
        optimizer = LSRAdam(model.parameters(), lr=args.lr*((1.+4.*args.sigma)**(0.25)), 
                           betas=(args.beta1, args.beta2),
                           weight_decay=args.weight_decay, 
                           sigma=args.sigma)
    elif args.optimizer.lower() == 'srsgd':
        iter_count = 1
        optimizer = SGD_Adaptive(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, iter_count=iter_count, restarting_iter=args.restart_schedule[0])
    elif args.optimizer.lower() == 'sradam':
        iter_count = 1
        optimizer = SRNAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), iter_count=iter_count, weight_decay=args.weight_decay, restarting_iter=args.restart_schedule[0]) 
    elif args.optimizer.lower() == 'sradamw':
        iter_count = 1
        optimizer = SRAdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), iter_count=iter_count, weight_decay=args.weight_decay, warmup = 0, restarting_iter=args.restart_schedule[0]) 
    elif args.optimizer.lower() == 'srradam':
        #NOTE: need to double-check this
        iter_count = 1
        optimizer = SRRAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), iter_count=iter_count, weight_decay=args.weight_decay, warmup = 0, restarting_iter=args.restart_schedule[0]) 
    
    schedule_index = 1
    # Resume
    title = 'ImageNet-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_top1 = checkpoint['best_top1']
        best_top5 = checkpoint['best_top5']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        iter_count = optimizer.param_groups[0]['iter_count']
        schedule_index = checkpoint['schedule_index']
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)  
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Top1', 'Valid Top1', 'Train Top5', 'Valid Top5'])
        
    logger.file.write('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))


    if args.evaluate:
        logger.file.write('\nEvaluation only')
        test_loss, test_top1, test_top5 = test(val_loader, model, criterion, start_epoch, use_cuda, logger)
        logger.file.write(' Test Loss:  %.8f, Test Top1:  %.2f, Test Top5: %.2f' % (test_loss, test_top1, test_top5))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        if args.optimizer.lower() == 'srsgd':
            if epoch in args.schedule:
                optimizer = SGD_Adaptive(model.parameters(), lr=args.lr * (args.gamma**schedule_index), weight_decay=args.weight_decay, iter_count=iter_count, restarting_iter=args.restart_schedule[schedule_index])
                schedule_index += 1
                
        elif args.optimizer.lower() == 'sradam':
            if epoch in args.schedule:
                optimizer = SRNAdam(model.parameters(), lr=args.lr * (args.gamma**schedule_index), betas=(args.beta1, args.beta2), iter_count=iter_count, weight_decay=args.weight_decay, restarting_iter=args.restart_schedule[schedule_index]) 
                schedule_index += 1
                
        elif args.optimizer.lower() == 'sradamw':
            if epoch in args.schedule:
                optimizer = SRAdamW(model.parameters(), lr=args.lr * (args.gamma**schedule_index), betas=(args.beta1, args.beta2), iter_count=iter_count, weight_decay=args.weight_decay, warmup = 0, restarting_iter=args.restart_schedule[schedule_index])
                schedule_index += 1
                
        elif args.optimizer.lower() == 'srradam':
            if epoch in args.schedule:
                optimizer = SRRAdam(model.parameters(), lr=args.lr * (args.gamma**schedule_index), betas=(args.beta1, args.beta2), iter_count=iter_count, weight_decay=args.weight_decay, warmup = 0, restarting_iter=args.restart_schedule[schedule_index])
                schedule_index += 1
            
        else:
            adjust_learning_rate(optimizer, epoch)

        logger.file.write('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        
        if args.optimizer.lower() == 'srsgd' or args.optimizer.lower() == 'sradam' or args.optimizer.lower() == 'sradamw' or args.optimizer.lower() == 'srradam':
            train_loss, train_top1, train_top5, iter_count = train(train_loader, model, criterion, optimizer, epoch, use_cuda, logger)
        else:
            train_loss, train_top1, train_top5 = train(train_loader, model, criterion, optimizer, epoch, use_cuda, logger)

        test_loss, test_top1, test_top5 = test(val_loader, model, criterion, epoch, use_cuda, logger)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_top1, test_top1, train_top5, test_top5])

        writer.add_scalars('train_loss', {args.model_name: train_loss}, epoch)
        writer.add_scalars('test_loss', {args.model_name: test_loss}, epoch)
        writer.add_scalars('train_top1', {args.model_name: train_top1}, epoch)
        writer.add_scalars('test_top1', {args.model_name: test_top1}, epoch)
        writer.add_scalars('train_top5', {args.model_name: train_top5}, epoch)
        writer.add_scalars('test_top5', {args.model_name: test_top5}, epoch)

        # save model
        is_best = test_top1 > best_top1
        best_top1 = max(test_top1, best_top1)
        best_top5 = max(test_top5, best_top5)
        save_checkpoint({
                'epoch': epoch + 1,
                'schedule_index': schedule_index,
                'state_dict': model.state_dict(),
                'top1': test_top1,
                'top5': test_top5,
                'best_top1': best_top1,
                'best_top5': best_top5,
                'optimizer' : optimizer.state_dict(),
            }, is_best, epoch, checkpoint=args.checkpoint)
        
        # reset DALI iterators
        train_loader.reset()
        val_loader.reset()
        
    logger.file.write('Best top1: %f'%best_top1)
    logger.file.write('Best top5: %f'%best_top5)
    
    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best top1: %f'%best_top1)
    print('Best top5: %f'%best_top5)
    
    with open("./all_results_imagenet.txt", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write("%s\n"%args.checkpoint)
        f.write("best_top1 %f, best_top5 %f\n\n"%(best_top1,best_top5))
        fcntl.flock(f, fcntl.LOCK_UN)

def train(train_loader, model, criterion, optimizer, epoch, use_cuda, logger):
    global batch_time_global, data_time_global
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    train_loader_len = int(train_loader._size / args.train_batch) + 1
    bar = Bar('Processing', max=train_loader_len)
    for batch_idx, data in enumerate(train_loader):
        # measure data loading time
        data_time_lap = time.time() - end
        data_time.update(data_time_lap)
        if epoch > 0:
            data_time_global.update(data_time_lap)
        
        inputs = data[0]["data"]
        targets = data[0]["label"].squeeze().cuda().long()

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(to_python_float(loss.data), inputs.size(0))
        top1.update(to_python_float(prec1), inputs.size(0))
        top5.update(to_python_float(prec5), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # for restarting
        if args.optimizer.lower() == 'srsgd' or args.optimizer.lower() == 'sradam' or args.optimizer.lower() == 'sradamw' or args.optimizer.lower() == 'srradam':
            iter_count, iter_total = optimizer.update_iter()

        # measure elapsed time
        batch_time_lap = time.time() - end
        batch_time.update(batch_time_lap)
        if epoch > 0:
            batch_time_global.update(batch_time_lap)
        end = time.time()

        # plot progress
        bar.suffix  = '(Epoch {epoch}, {batch}/{size}) Data: {data:.3f}s/{data_global:.3f}s | Batch: {bt:.3f}s/{bt_global:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    epoch=epoch,
                    batch=batch_idx + 1,
                    size=train_loader_len,
                    data=data_time.val,
                    data_global=data_time_global.avg,
                    bt=batch_time.val,
                    bt_global=batch_time_global.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
        logger.file.write(bar.suffix)
    bar.finish()
    if args.optimizer.lower() == 'srsgd' or args.optimizer.lower() == 'sradam' or args.optimizer.lower() == 'sradamw' or args.optimizer.lower() == 'srradam':
        return (losses.avg, top1.avg, top5.avg, iter_count)
    else:
        return (losses.avg, top1.avg, top5.avg)

def test(val_loader, model, criterion, epoch, use_cuda, logger):
    global best_top1, best_top5

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    
    val_loader_len = int(val_loader._size / args.test_batch)
    bar = Bar('Processing', max=val_loader_len)
    for batch_idx, data in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        inputs = data[0]["data"]
        targets = data[0]["label"].squeeze().cuda().long()

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(to_python_float(loss.data), inputs.size(0))
        top1.update(to_python_float(prec1), inputs.size(0))
        top5.update(to_python_float(prec5), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '(Epoch {epoch}, {batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    epoch=epoch,
                    batch=batch_idx + 1,
                    size=val_loader_len,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        logger.file.write(bar.suffix)
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, top5.avg)
        
def save_checkpoint(state, is_best, epoch, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    next_epoch = epoch + 1
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
    if next_epoch in args.schedule:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_epoch_%i.pth.tar'%epoch))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
    # writer.close()
