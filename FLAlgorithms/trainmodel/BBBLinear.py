import sys
sys.path.append("..")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
from torch.nn import Parameter
from FLAlgorithms.trainmodel.layers.misc import ModuleWrapper
from torch.distributions.normal import Normal

global_sigma_size = 1e-3

def softplus_inverse(y):
    return torch.log1p(torch.exp(y) - 1)

def getNormalDist(mu, rho):
    # sigma = torch.log1p(torch.exp(rho))
    sigma = F.softplus(rho)
    return dist.Normal(mu, sigma * global_sigma_size)

class NormalParameter(nn.Module):
    def __init__(self, parameter):
        super(NormalParameter, self).__init__()
        self.parameter = Parameter(parameter)

    def forward(self):
        return self.parameter

class BParameter(ModuleWrapper):
    def __init__(self, parameter):
        super(BParameter, self).__init__()
        self.sample = True
        self.parameter_mu = Parameter(parameter)
        self.parameter_rho = Parameter(torch.empty(parameter.shape))
        self.global_sigma_size = global_sigma_size
        self.reset_parameters()

    def reset_parameters(self):
        self.parameter_rho.data.normal_(0.0, 1e-3)

    def forward(self):
        if self.sample:
            W_eps = torch.empty(self.parameter_mu.size()).normal_(0, 1)
            # parameter_sigma = torch.log1p(torch.exp(self.parameter_rho))
            parameter_sigma = F.softplus(self.parameter_rho)
            # weight = self.W_mu + W_eps * self.W_rho * global_size
            parameter = self.parameter_mu + W_eps * (parameter_sigma * self.global_sigma_size)

        else:
            parameter = self.parameter_mu

        return parameter

    def dist(self):
        # self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        # parameter_sigma = torch.log1p(torch.exp(self.parameter_rho))
        parameter_sigma = F.softplus(self.parameter_rho)
        return [dist.Normal(self.parameter_mu, parameter_sigma)]

class FFGLinear(ModuleWrapper):
    def __init__(self, in_features, out_features, bias=True, priors=None):
        super(FFGLinear, self).__init__()
        self.sample = True
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if priors is None:
            priors = {
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (0, 1e-6),
            }
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(torch.empty((out_features, in_features), device=self.device))
        self.W_rho = Parameter(torch.empty((out_features, in_features), device=self.device))

        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_features), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_features), device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.global_sigma_size = global_sigma_size
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_mu, a=math.sqrt(5))
        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_mu)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_mu, -bound, bound)

        self.W_rho.data.normal_(*self.posterior_rho_initial)
        if self.use_bias:
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, input):
        if self.sample:
            W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
            # W_sigma = torch.log1p(torch.exp(self.W_rho))
            W_sigma = F.softplus(self.W_rho)
            weight = self.W_mu + W_eps * W_sigma * self.global_sigma_size

            if self.use_bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
                # bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias_sigma = F.softplus(self.bias_rho)
                bias = self.bias_mu + bias_eps * bias_sigma  * self.global_sigma_size
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return F.linear(input, weight, bias)

    def dist(self):
        # W_sigma = torch.log1p(torch.exp(self.W_rho))
        W_sigma = F.softplus(self.W_rho)
        if (self.use_bias):
            # bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_sigma = F.softplus(self.bias_rho)
            return [dist.Normal(self.W_mu, W_sigma), dist.Normal(self.bias_mu, bias_sigma)]
        else:
            return [dist.Normal(self.W_mu, W_sigma)]