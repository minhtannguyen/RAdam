import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import logging
import os
import shutil
import time
from datetime import timedelta
import sys

import numpy as np
import torch
from torch.autograd import Variable
import numpy as np
import torch.distributed as dist

from genotypes import PRIMITIVES
from operations import *

import torch.nn.functional as F
from tensorboardX import SummaryWriter


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class ExpMovingAvgrageMeter(object):

    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.avg = 0

    def update(self, val):
        self.avg = (1. - self.momentum) * self.avg + self.momentum * val


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def count_parameters_in_M(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = torch.cuda.FloatTensor(
            x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x = x / keep_prob
        x = x * mask
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class ClassErrorMeter(object):
    def __init__(self):
        super(ClassErrorMeter, self).__init__()
        self.class_counter = {}

    def add(self, output, target):
        _, pred = output.max(dim=1)

        target = list(target.cpu().numpy())
        pred = list(pred.cpu().numpy())

        for t, p in zip(target, pred):
            if t not in self.class_counter:
                self.class_counter[t] = {'num': 0, 'correct': 0}
            self.class_counter[t]['num'] += 1
            if t == p:
                self.class_counter[t]['correct'] += 1

    def value(self, method):
        print('Error type: ', method)
        if method == 'per_class':
            mean_accuracy = 0
            for t in self.class_counter:
                class_accuracy = float(self.class_counter[t]['correct']) / \
                    self.class_counter[t]['num']
                mean_accuracy += class_accuracy
            mean_accuracy /= len(self.class_counter)
            output = mean_accuracy * 100
        elif method == 'overall':
            num_total, num_correct = 0, 0
            for t in self.class_counter:
                num_total += self.class_counter[t]['num']
                num_correct += self.class_counter[t]['correct']
            output = float(num_correct) / num_total * 100
        return [100 - output]


def sample_gumbel(shape, eps=1e-20):
    U = torch.Tensor(shape).uniform_(0, 1).cuda()
    sample = -(torch.log(-torch.log(U + eps) + eps))
    return sample


def gumbel_softmax_sample_original(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def logsumexp(logits, dim):
    mx = torch.max(logits, dim, keepdim=True)[0]
    return torch.log(torch.sum(torch.exp(logits - mx), dim=dim, keepdim=True)) + mx


def gumbel_softmax_sample_improved(logits, temperature):
    def gsm(rho, q):
        return F.softmax((-torch.log(rho + 1e-20) + torch.log(q + 1e-20)) / temperature, dim=-1)
    q = F.softmax(logits, dim=-1)
    U = torch.Tensor(q.size()).uniform_(0, 1).cuda()
    U = torch.clamp(U, 1e-15, 1. - 1e-15)
    log_U = torch.log(U)
    rho = log_U / (torch.sum(log_U, dim=-1, keepdim=True))
    return gsm(rho.detach() - q + q.detach(), q.detach())


def gumbel_softmax_sample_rebar(logits, temperature):
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    q = F.softmax(logits, dim=-1)
    u = torch.Tensor(q.size()).uniform_(0, 1).cuda()
    u = torch.clamp(u, 1e-3, 1.-1e-3)

    # draw gsm samples
    z = logits - torch.log(- torch.log(u))
    gsm = F.softmax(z / temperature, dim=-1)

    # compute the correction term for conditional samples
    # see REBAR: https://arxiv.org/pdf/1703.07370.pdf
    k = torch.argmax(z, dim=-1, keepdim=True)
    # get v from u
    u_k = u.gather(-1, k)
    q_k = q.gather(-1, k)
    # This can cause numerical problems, better to work with log(v_k) = u_k / q_k
    # v_k = torch.pow(u_k, 1. / q_k)
    # v.scatter_(-1, k, v_k)
    log_vk = torch.log(u_k) / q_k
    log_v = torch.log(u) - q * log_vk

    # assume k and v are constant
    k = k.detach()
    log_vk = log_vk.detach()
    log_v = log_v.detach()
    g_hat = - torch.log(-log_v/q - log_vk)
    g_hat.scatter(-1, k, -torch.log(- log_vk))
    gsm1 = F.softmax(g_hat / temperature, dim=-1)

    return gsm - gsm1 + gsm1.detach()


def gumbel_softmax_sample(logits, temperature, gsm_type='improved'):
    if gsm_type == 'improved':
        return gumbel_softmax_sample_improved(logits, temperature)
    elif gsm_type == 'original':
        return gumbel_softmax_sample_original(logits, temperature)
    elif gsm_type == 'rebar':
        return gumbel_softmax_sample_rebar(logits, temperature)


def plot_alphas(alpha, display=True, title='', savename=''):
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(7)

    alpha = alpha.data.cpu().numpy()

    num_edges = alpha.shape[0]
    ops = PRIMITIVES

    ax.xaxis.tick_top()
    plt.imshow(alpha, vmin=0, vmax=1)
    plt.xticks(range(len(ops)), ops)
    plt.xticks(rotation=30)
    plt.yticks(range(num_edges), range(1, num_edges+1))
    for i in range(num_edges):
        for j in range(len(ops)):
            val = alpha[i][j]
            val = '%.4f' % (val)
            ax.text(j, i, val, va='center',
                    ha='center', color='white', fontsize=8)

    plt.colorbar()
    plt.tight_layout()
    fig.suptitle(title, fontsize=16, fontweight='bold')

    if savename:
        plt.savefig(savename)
    if display:
        plt.show()
    else:
        return fig


def plot_alphas_paired_input(alpha, display=True, title='', savename=''):
    fig, ax = plt.subplots(1, 2)
    fig.set_figheight(6)
    fig.set_figwidth(13)

    for idx, k in enumerate(sorted(alpha.keys())):
        prob = [F.softmax(a, dim=-1) for a in alpha[k]]
        if k == 'input':
            selector = torch.cat(prob, 1)
        elif k == 'combiner':
            selector = torch.cat(prob, 0).view(len(alpha[k]), -1)
        else:
            selector = torch.cat(prob, 0)

        selector = selector.data.cpu().numpy()
        im = ax[idx].imshow(selector, vmin=0, vmax=1)
        # ax[idx].set_title(k)
        ax[idx].set_xlabel(k)
        xticks = get_xticks(alpha[k], k)
        plt.sca(ax[idx])
        ax[idx].xaxis.tick_top()
        plt.xticks(range(len(xticks)), xticks)
        plt.xticks(ha='left', rotation=30, fontsize=8)
        plt.yticks(range(selector.shape[0]), range(1, selector.shape[0] + 1))

    cbaxes = fig.add_axes([0.05, 0.2, 0.01, 0.6])
    fig.colorbar(im, cax=cbaxes)
    cbaxes.yaxis.set_ticks_position('left')

    if savename:
        plt.savefig(savename)
    if display:
        plt.show()
    else:
        return fig


def get_xticks(alpha, key):
    if key == 'input':
        xticks = []
        for i in range(len(alpha)):
            for j in range(i+2):
                xticks.append(j)
    elif key == 'op':
        xticks = PRIMITIVES
    elif key == 'activation':
        xticks = ACTIVATIONS.keys()
    elif key == 'combiner':
        xticks = COMBINERS.keys()
    else:
        raise NotImplementedError

    return xticks


def generate_paired_indices(step):
    indices = []
    for i in range(step + 2):
        indices.append((i, i))

    for i in range(step + 2):
        for j in range(i + 1, step + 2):
            indices.append((i, j))

    return indices


class Logger(object):
    def __init__(self, rank, save):
        self.rank = rank
        if self.rank == 0:
            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(save, 'log.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(fh)
            self.start_time = time.time()

    def info(self, string, *args):
        if self.rank == 0:
            elapsed_time = time.time() - self.start_time
            elapsed_time = time.strftime(
                '(Elapsed: %H:%M:%S) ', time.gmtime(elapsed_time))
            if isinstance(string, str):
                string = elapsed_time + string
            else:
                logging.info(elapsed_time)
            logging.info(string, *args)


class Writer(object):
    def __init__(self, rank, save):
        self.rank = rank
        if self.rank == 0:
            try:
                self.writer = SummaryWriter(log_dir=save, flush_secs=20)
            except:
                self.writer = SummaryWriter(logdir=save, flush_secs=20)

    def add_scalar(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_scalar(*args, **kwargs)

    def add_figure(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_figure(*args, **kwargs)

    def flush(self):
        if self.rank == 0:
            self.writer.flush()


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def get_memory_usage(device=None):
    try:
        memory_cached = torch.cuda.max_memory_cached(device) * 1e-9
        memory_alloc = torch.cuda.max_memory_allocated(device) * 1e-9
        torch.cuda.reset_max_memory_allocated(device)
        torch.cuda.reset_max_memory_cached(device)
    except Exception:
        memory_cached, memory_alloc = 0., 0.
    return memory_cached, memory_alloc


def is_parametric(primitive):
    if primitive in {'none', 'skip_connect', 'max_pool_3x3', 'avg_pool_3x3'}:
        return False
    elif primitive in {'sep_conv_3x3', 'sep_conv_5x5', 'sep_conv_7x7', 'dil_conv_3x3', 'dil_conv_5x5'}:
        return True
    else:
        raise KeyError('primitive %s is not in the list' % primitive)


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.half()


if __name__ == '__main__':
    avg_meter = ExpMovingAvgrageMeter(momentum=0.9)
    for i in range(100):
        avg_meter.update(i)
        print(avg_meter.avg)
