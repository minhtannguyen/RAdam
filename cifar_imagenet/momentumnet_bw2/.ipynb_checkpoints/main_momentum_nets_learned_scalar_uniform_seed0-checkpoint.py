# -*- coding: utf-8 -*-
"""
Test momentum nets
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

import numpy as np
import numpy.matlib

import random

from momentum_nets_learned_scalar_uniform import *
from utils import *


from sgd import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="6"


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    global best_acc
    best_acc = 0
    start_epoch = 0
    
    random.seed(0)
    torch.manual_seed(0)
    if use_cuda:
        torch.cuda.manual_seed_all(0)
    
    #--------------------------------------------------------------------------
    # Load the Cifar10 data.
    #--------------------------------------------------------------------------
    print('==> Preparing data...')
    root = '../data_Cifar10'
    download = True
    
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    
    train_set = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    test_set = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    
    # Convert the data into appropriate torch format.
    kwargs = {'num_workers':1, 'pin_memory':True}
    
    batchsize_test = 20
    print('Batch size of the test set: ', batchsize_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batchsize_test,
                                              shuffle=False, **kwargs
                                             )
    
    batchsize_train = 128
    print('Batch size of the train set: ', batchsize_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batchsize_train,
                                               shuffle=True, **kwargs
                                              )
    
    batchsize_train = len(train_set)
    print('Total training (known) batch number: ', len(train_loader))
    print('Total testing batch number: ', len(test_loader))
    
    #--------------------------------------------------------------------------
    # Build the model
    #--------------------------------------------------------------------------
    #net = momentum_net20().cuda()
    #net = momentum_net56().cuda()
    #net = momentum_net110().cuda()
    #net = momentum_net164().cuda()
    net = momentum_net290().cuda()
    
    lr = 0.1 # Changed
    weight_decay = 5e-4
    
    #global iter_count
    iter_count = 1
    optimizer = SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    
    nepoch = 200
    for epoch in range(nepoch):
        print('Epoch ID: ', epoch)
        if epoch == 80:
            optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        elif epoch == 120:
            optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        elif epoch == 160:
            optimizer = SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        
        correct = 0; total = 0; train_loss = 0
        net.train()
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x, target = Variable(x.cuda()), Variable(target.cuda())
            score, loss = net(x, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(score.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
            
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        #----------------------------------------------------------------------
        # Testing
        #----------------------------------------------------------------------
        test_loss = 0; correct = 0; total = 0
        net.eval()
        for batch_idx, (x, target) in enumerate(test_loader):
            x, target = Variable(x.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
            score, loss = net(x, target)
            
            test_loss += loss.item() #data[0]
            _, predicted = torch.max(score.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        #----------------------------------------------------------------------
        # Save the checkpoint
        #----------------------------------------------------------------------
        #acc = 100.*correct/total
        acc = (1.*correct.data.cpu().numpy())/10000.
        if acc > best_acc:
            print('Saving model...')
            state = {
                'net': net,
                'acc': acc,
                'epoch': epoch,
            }
#             if not os.path.isdir('checkpoint_Cifar'):
#                 os.mkdir('checkpoint_Cifar')
#             torch.save(state, './checkpoint_Cifar/ckpt.t7')
            best_acc = acc
    
    print('The best acc: ', best_acc)
