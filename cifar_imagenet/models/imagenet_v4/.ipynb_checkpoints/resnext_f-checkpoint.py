"""ResNeXt implementation (https://arxiv.org/abs/1611.05431)."""
import math
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class ResNeXtBottleneck(nn.Module):
  """ResNeXt Bottleneck Block type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)."""
  expansion = 4

    def __init__(self,
               inplanes,
               planes,
               cardinality,
               base_width,
               stride=1,
               downsample=None,
               ncycles=1):
        super(ResNeXtBottleneck, self).__init__()

        dim = int(math.floor(planes * (base_width / 64.0)))

        self.conv_reduce = nn.Conv2d(
            inplanes,
            dim * cardinality,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        
        self.bn_reduce = []
        for i in range(ncycles):
            self.bn_reduce.append(nn.BatchNorm2d(dim * cardinality))

        self.conv_conv = nn.Conv2d(
            dim * cardinality,
            dim * cardinality,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        
        self.bn = []
        for i in range(ncycles):
            self.bn.append(nn.BatchNorm2d(dim * cardinality))

        self.conv_expand = nn.Conv2d(
            dim * cardinality,
            planes * 4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        
        self.bn_expand = []
        for i in range(ncycles):
            self.bn_expand.append(nn.BatchNorm2d(planes * 4))

        self.downsample = downsample

    def forward(self, x):
        x = x[0]
        bn_indx = x[1]
        
        residual = x

        bottleneck = self.conv_reduce(x)
        bottleneck = F.relu(self.bn_reduce[bn_indx](bottleneck), inplace=True)

        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.relu(self.bn[bn_indx](bottleneck), inplace=True)

        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand[bn_indx](bottleneck)

        if self.downsample is not None:
            residual = self.downsample([x, bn_indx])
            residual = residual[0]

        return [F.relu(residual + bottleneck, inplace=True), bn_indx]

class Downsample(nn.Module):
  """ResNeXt Bottleneck Block type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)."""
  expansion = 4

    def __init__(self,
               inplanes,
               planes,
               stride=1,
               ncycles=1):
        super(Downsample, self).__init__()
        
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.ncycles = ncycles
        
        self.conv = nn.Conv2d(
              self.inplanes,
              self.planes,
              kernel_size=1,
              stride=self.stride,
              bias=False)
        
        self.bn = []
        for i in range(self.ncycles):
            self.bn.(nn.BatchNorm2d(self.planes))
        
    def forward(self, x):
        out = x[0]
        bn_indx = x[1]
        out = self.conv(out)
        out = self.bn[bn_indx](out)
        
        return [out, bn_indx]


class CifarResNeXt(nn.Module):
  """ResNext optimized for the Cifar dataset, as specified in https://arxiv.org/pdf/1611.05431.pdf."""

  def __init__(self, block, depth, cardinality, base_width, num_classes, ncycles):
    super(CifarResNeXt, self).__init__()

    # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
    assert (depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
    layer_blocks = (depth - 2) // 9

    self.cardinality = cardinality
    self.base_width = base_width
    self.num_classes = num_classes
    self.ncycles=ncycles

    self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    
    self.bn_1 = []
    for i in range(ncycles):
        self.bn_1.append(nn.BatchNorm2d(64))

    self.inplanes = 64
    self.stage_1 = self._make_layer(block, 64, layer_blocks, 1, ncycles=self.ncycles)
    self.stage_2 = self._make_layer(block, 128, layer_blocks, 2, ncycles=self.ncycles)
    self.stage_3 = self._make_layer(block, 256, layer_blocks, 2, ncycles=self.ncycles)
    self.avgpool = nn.AvgPool2d(8)
    self.classifier = nn.Linear(256 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal(m.weight)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, ncycles=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
        downsample = Downsample(inplanes=self.inplanes,
               planes=planes * block.expansion,
               stride=stride,
               ncycles=ncycles)
    layers = []
    layers.append(
        block(self.inplanes, planes, self.cardinality, self.base_width, stride,
              downsample, ncycles=ncycles))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(
          block(self.inplanes, planes, self.cardinality, self.base_width, ncycles=ncycles))

    return nn.Sequential(*layers)

  def forward_once(self, x, bn_indx=1):
    x = self.conv_1_3x3(x)
    x = F.relu(self.bn_1[bn_indx](x), inplace=True)
    
    x = [x, bn_indx]
    x = self.stage_1(x)
    x = self.stage_2(x)
    x = self.stage_3(x)
    x = x[0]
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return self.classifier(x)

  def forward(self, x, cycles, coeff, mse_parameter=0.1, kl_parameter=0.1, tv_parameter=0.1):
    x_cur = x
    for i_cycle in range(cycles):
        # feedforward
        output = self.forward_once(x_cur, i_cycle)
        # Xentropy loss (using logits)
        cat_loss = self.Xentropy(output, target)  
        # label smoothing  
        softmax = F.softmax(output, dim=1)
        kl_loss = -torch.mean(torch.sum(torch.log(10.0*softmax + 1e-8)*softmax, dim=1))
        # feedback: using softmax                
        softmax_out = - torch.sum(F.log_softmax(output, dim=1)) / self.categories
        grad_outs = torch.ones_like(softmax_out)
        grad = torch.autograd.grad(softmax_out, x_cur, grad_outputs=grad_outs,
                         retain_graph=True, create_graph=True,
                         allow_unused=True)[0]   
        # gradient augmented image
        x_cur = x_cur + self.step * grad
        # Reconstruction loss in L2
        rec_loss = F.mse_loss(x_cur, data)
        # TV loss
        tv_loss = self.TV(x_cur)


def resnext29(num_classes=10, cardinality=4, base_width=32):
  model = CifarResNeXt(ResNeXtBottleneck, 29, cardinality, base_width,
                       num_classes)
  return model
