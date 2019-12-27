import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

def mulvt(v,t):
##################################
## widely used in binary search ##
## v is batch_size by 1         ##
## t is batch_size by any dim   ##
##################################
    batch_size, other_dim = t.size()[0], t.size()[1:]
    len_dim = len(other_dim)-1
    for i in range(len_dim):
        v = v.unsqueeze(len(v.size()))
    v = v.expand(t.size())
    return v*t    
    
def reduce_sum(t,axis):
    dim = t.size()

class FGSM(object):
    def __init__(self,model):
        self.model = model

    def get_loss(self,xi,label_or_target,TARGETED):
        criterion = nn.CrossEntropyLoss()
        output = self.model.predict(xi)
        #print(output, label_or_target)
        loss = criterion(output, label_or_target)
        #print(loss)
        #print(c.size(),modifier.size())
        return loss
    
    def pgd_whitebox(self, X, y, epsilon=0.031, num_steps=20, step_size=0.003):
        
#         y = Variable(y.cuda())
#         X = Variable(X.cuda(), requires_grad=True)
        
#         out = self.model.predict(X)
#         err = (out.data.max(1)[1] != y.data).float().sum()
        X = Variable(X.data, requires_grad=True).cuda()
        X_pgd = Variable(X.data, requires_grad=True).cuda()
        y = Variable(y.cuda())
        
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
        #X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True).cuda()
        X_pgd = Variable(X_pgd.data, requires_grad=True).cuda()

        for _ in range(num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()

            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(self.model.predict(X_pgd), y)
            loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
#         err_pgd = (self.model(X_pgd).data.max(1)[1] != y.data).float().sum()
#         print('err pgd (white-box): ', err_pgd)
        return X_pgd

    def i_fgsm(self, input_xi, label_or_target, eta, bound=(0,1), TARGETED=False):
       
        yi = Variable(label_or_target.cuda())
        x_adv = Variable(input_xi.cuda(), requires_grad=True)
        for it in range(20):
            error = self.get_loss(x_adv,yi,TARGETED)
            # print(error.data[0]) 
            self.model.get_gradient(error)
            #print(gradient)
            x_adv.grad.sign_()
            if TARGETED:
                x_adv.data = x_adv.data - eta* x_adv.grad 
                x_adv.data = torch.clamp(x_adv.data, bound[0], bound[1])
            else:
                x_adv.data = x_adv.data + eta* x_adv.grad
                x_adv.data = torch.clamp(x_adv.data, bound[0], bound[1])
            #x_adv = Variable(x_adv.data, requires_grad=True)
            #error.backward()
        return x_adv

    def fgsm(self, input_xi, label_or_target, eta, bound=(0,1), TARGETED=False):
       
        yi = Variable(label_or_target.cuda())
        x_adv = Variable(input_xi.cuda(), requires_grad=True)

        error = self.get_loss(x_adv,yi,TARGETED)
        self.model.get_gradient(error)
        #print(gradient)
        x_adv.grad.sign_()
        if TARGETED:
            x_adv.data = x_adv.data - eta* x_adv.grad 
            x_adv.data = torch.clamp(x_adv.data, bound[0], bound[1])
        else:
            x_adv.data = x_adv.data + eta* x_adv.grad
            x_adv.data = torch.clamp(x_adv.data, bound[0], bound[1])
            #x_adv = Variable(x_adv.data, requires_grad=True)
            #error.backward()
        return x_adv 

    def __call__(self, input_xi, label_or_target, eta=0.01, TARGETED=False):
        adv = self.i_fgsm(input_xi, label_or_target, eta, TARGETED)
        return adv   


        