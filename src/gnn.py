import sys
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
 
from torch_geometric.nn import GCNConv # My part

# Whether use adjoint method or not.
adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class ODEFunc(nn.Module):

    # currently requires in_features = out_features
    def __init__(self, in_features, out_features, opt, adj, deg):
        super(ODEFunc, self).__init__()
        self.opt = opt
        self.adj = adj
        self.x0 = None
        self.nfe = 0
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = opt['alpha']
        self.alpha_train = nn.Parameter(self.alpha*torch.ones(adj.shape[1]))

        self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
        self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)

    def forward(self, t, x):
        self.nfe +=1

        alph = F.sigmoid(self.alpha_train).unsqueeze(dim=1)
        ax = torch.spmm(self.adj, x)
        f = alph * 0.5 * (ax-x) + self.x0
        return f


#Define my own Custom ODEFunc   
class Custom_ODEFunc(nn.Module):

    # currently requires in_features = out_features
    def __init__(self, in_features, out_features, opt, adj, deg):
        super(Custom_ODEFunc, self).__init__()
        self.opt = opt
        self.adj = adj
        self.x0 = None
        self.nfe = 0
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = opt['alpha']
        self.alpha_train = nn.Parameter(self.alpha*torch.ones(adj.shape[1]))

        self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
        self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)

    def forward(self, t, x):
        self.nfe +=1

        alph = F.leaky_relu(self.alpha_train).unsqueeze(dim=1)
        ax = torch.spmm(self.adj, x)
        f = alph * 0.6 * (ax-x) + self.x0
        return f

class ODEblock(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0,1])):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc
        self.nfe = 0

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x):
        self.nfe += 1

        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t)[1]
        return z

    def __repr__(self):
        return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
               + ")"


# Define the GNN model.
class GNN(nn.Module):
    def __init__(self, opt, adj, deg, time):
        super(GNN, self).__init__()
        self.opt = opt
        self.adj = adj
        self.T = time

        self.m1 = nn.Linear(opt['num_feature'], opt['hidden_dim'])

        self.odeblock = ODEblock(ODEFunc(2*opt['hidden_dim'], 2*opt['hidden_dim'], opt, adj, deg), t=torch.tensor([0,self.T]))

        self.m2 = nn.Linear(opt['hidden_dim'], opt['num_class'])

        if opt['cuda']:
            self.cuda()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def forward(self, x):
        # Encode each node based on its feature.
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)

        # Solve the initial value problem of the ODE.
        c_aux = torch.zeros(x.shape).cuda()
        x = torch.cat([x,c_aux], dim=1)
        self.odeblock.set_x0(x)

        z = self.odeblock(x)
        z = torch.split(z, x.shape[1]//2, dim=1)[0]

        # Activation.
        z = F.relu(z)

        # Dropout.
        z = F.dropout(z, self.opt['dropout'], training=self.training)

        # Decode each node embedding to get node label.
        z = self.m2(z)
        return z




# Define my own GNN model.
class Custom_GNN(nn.Module):
    def __init__(self, opt, adj, deg, time):
        super(Custom_GNN, self).__init__()
        self.opt = opt
        self.adj = adj
        self.T = time

        self.conv1 = GCNConv(opt['num_feature'], opt['hidden_dim'])  # First GCN layer
        
        self.odeblock = ODEblock(Custom_ODEFunc(3*opt['hidden_dim'], 3*opt['hidden_dim'], opt, adj, deg), t=torch.tensor([0,self.T]))

        self.conv2 = GCNConv(opt['hidden_dim'], opt['num_class'])  # Second GCN layer

  

        if opt['cuda']:
            self.cuda()

    def reset(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
      

    def forward(self, x):
        # Encode each node based on its feature.
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        

        # Coalesce the adjacency matrix.
        self.adj = self.adj.coalesce()
     
        # Apply the first GCN layer.
        x = self.conv1(x, self.adj)
  

        # Solve the initial value problem of the ODE.
        c_aux = torch.zeros(x.shape).cuda()
        x = torch.cat([x,c_aux], dim=1)
        self.odeblock.set_x0(x)

        z = self.odeblock(x)
        z = torch.split(z, x.shape[1]//2, dim=1)[0]

        # Activation.
        z = F.relu(z)

        # Dropout.
        z = F.dropout(z, self.opt['dropout'], training=self.training)

        # Decode each node embedding to get node label.
        z = self.conv2(z, self.adj)



        return z

