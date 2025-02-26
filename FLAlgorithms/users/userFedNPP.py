from scipy.stats import poisson, uniform, gamma, dirichlet, norm
# from scipy.linalg import solve_triangular
# from scipy.special import digamma, logsumexp, gammaln
import time
import copy
import matplotlib.pyplot as plt
import sys
import numpy as np
from torch import distributions as dist
import torch
import torch.nn as nn
from scipy.integrate import quadrature
from sys import getsizeof, stderr
from torch.distributions.kl import kl_divergence
from itertools import chain
from collections import deque
from utils.utils import cov_func, logcosh, sigma_f, exp_with_upperbound
from FLAlgorithms.trainmodel.DLMKernel import OneLayerMLP
from FLAlgorithms.trainmodel.BBDLMKernel import BBDLMKernel 
from FLAlgorithms.trainmodel.BBBLinear import NormalParameter, BParameter, getNormalDist
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class UserFedNPP(nn.Module):
    def __init__(self, client_idx, num_integration_points, num_inducing_points, train_data, args):
        super(UserFedNPP, self).__init__()
        self.args = args
        data = train_data['data'][client_idx]
        m_initial = train_data['m_data'][client_idx]
        cov_params_initial = train_data['cov_params_data'][client_idx]
        U_initial = train_data['inducing_U_data'][client_idx]
        Z_initial = train_data['inducing_location_data'][client_idx]
        self.data_range = train_data['sampling_range']
        self.num_iteration = args['max_epoch']
        self.global_param_dict = []

        print(self.data_range)
        self.T = self.data_range[1]
        self.num_integration_points = torch.tensor(num_integration_points)
        self.num_inducing_points = torch.tensor(num_inducing_points)
        self.data = data
        self.NumE = torch.tensor(len(data))
        self.noise = torch.tensor(0.01)

        # self.num_iteration = torch.tensor(100)
        if (args['is_bayesian_paramter'] == True):
            self.DPM_Kernel = BBDLMKernel(hidden_dim=128)
            self.mean_functions = BParameter(torch.zeros(1))

            if (cov_params_initial == None):
                self.cov_params = BParameter(torch.tensor([1.5, 10]))
            else:
                self.cov_params = BParameter(cov_params_initial)

            if Z_initial == None:
                # self.induced_location = BParameter(self.place_inducing_points())
                self.induced_location = self.place_inducing_points()
            else:
                # self.induced_location = BParameter(Z_initial)
                self.induced_location = Z_initial

        else:
            self.DPM_Kernel = OneLayerMLP(hidden_dim=128)
            self.mean_functions = NormalParameter(torch.zeros(1))

            if (cov_params_initial == None):
                self.cov_params = NormalParameter(torch.tensor([1.5, 10]))
            else:
                self.cov_params = NormalParameter(cov_params_initial)

            if Z_initial == None:
                # self.induced_location = NormalParameter(self.place_inducing_points())
                self.induced_location = self.place_inducing_points()
            else:
                # self.induced_location = NormalParameter(Z_initial)
                self.induced_location = Z_initial

        if (args['Kernel_DPM_Type'] == 'None'):
            self.DPM_Kernel = None

        if (m_initial == None):
            # Need Check
            # the prior of mm is gamma distribution with parameters alpha_mm_prior and beta_mm_prior
            # a in Eq. (23)
            # self.alpha_mm_prior = 0.1*self.Number of sequences in one client
            # self.beta_mm_prior = 0.1

            # mean = alpha/beta
            # variance = alpha/beta^2

            # in this way, mean of the prior is self.NUmber of sequences in one client
            # variance is 10*self.Number of sequences in one client
            num_data = 1
            self.alpha_mm_prior = torch.tensor(0.1 * num_data)
            # b in Eq. (23)
            self.beta_mm_prior = torch.tensor(0.1)

            self.mm = self.alpha_mm_prior/self.beta_mm_prior
            self.log_mm = torch.log1p(self.mm)
        else:
            self.alpha_mm_prior = torch.tensor(0.1 * m_initial)
            self.beta_mm_prior = torch.tensor(0.1)

            self.mm = self.alpha_mm_prior/self.beta_mm_prior
            self.log_mm = torch.log1p(self.mm)

        ##############################################################################

        # initialize inducing points values f_u, Sigma_f_u
        if U_initial == None:
            self.f_u = torch.normal(torch.zeros(self.num_inducing_points), torch.ones(self.num_inducing_points))
        else:
            self.f_u = U_initial # f_u is a list of tensors

        # refer to Eq. (31) - q(u)
        self.Sigma_f_u = torch.eye(self.num_inducing_points)
        self.Sigma_f_u_inv = torch.eye(self.num_inducing_points)

        ##############################################################################

        # K_zz
        self.Kzz = self.deep_cov_func_kernel(self.induced_location, self.induced_location, is_xx=True)+ self.noise*torch.eye(self.num_inducing_points)

        # # L = torch.linalg.cholesky(self.Kzz + self.noise*torch.eye(self.Kzz.shape[0]))
        # L = torch.linalg.cholesky(self.Kzz + self.noise*torch.eye(self.num_inducing_points))
        #
        # # L_inv = solve_triangular(L, torch.eye(L.shape[0]), lower = True, check_finite=False)
        # L_inv = torch.linalg.solve_triangular(L, torch.eye(L.shape[0]), upper = False)
        #
        # self.Kzz_inv = torch.matmul(L_inv.T, L_inv)

        L = torch.linalg.cholesky(self.Kzz)
        self.Kzz_inv = torch.cholesky_inverse(L)

        self.logdet_Kzz = 2. * torch.sum(torch.log1p(L.diagonal()))

        ##############################################################################
        ##############################################################################


        # self.logdet_Kzz = [2.*torch.sum(torch.log(L[k].diagonal())) for k in range(K)]

        # generate the integration points, and their corresponding kernel values
        self.place_integration_points()
        # Eq. (32) K_{z,t}
        self.k_u_int_points = self.deep_cov_func_kernel(self.induced_location, self.integration_location)
        # Eq. (33) kappa_int_points = K_{zz}^{-1}k_{z,t}
        self.kappa_int_points = torch.matmul(self.Kzz_inv, self.k_u_int_points)

        # self.origin_cov_f = origin_cov_f
        # self.origin_points = origin_points
        # self.origin_mu_f = origin_mu_f
        ##############################################################################

        # K_{z,data}
        self.k_u_X = self.deep_cov_func_kernel(self.induced_location, self.data)
        # K_{z,z}^{-1}K_{z,data}}
        self.kappa_X = torch.matmul(self.Kzz_inv, self.k_u_X)

        # f_X: mu of f_{data}
        # f2_X: var of f_{data}
        self.f_X, self.f2_X = self.predictive_posterior_GP_f(self.data, 'X')

        # f2_X: E[f_{data}^2]
        # E[f^2] = var(f) + E[f]^2
        self.f2_X = self.f2_X + self.f_X**2
        # self.f_X, var_f_X = [(data_i, 'X') for data_i in self.data]
        # self.f2_X = var_f_X + self.f_X**2

        # f_{int_points}: mu of f_{int_points}
        # f2_{int_points}: var of f_{int_points}
        self.f_int_points, self.f2_int_points = self.predictive_posterior_GP_f(self.integration_location, 'int_points')

        # f2_{int_points}: E[f_{int_points}^2]
        # E[f^2] = var(f) + E[f]^2
        self.f2_int_points = self.f2_int_points + self.f_int_points ** 2
        ##############################################################################

        self.optim = torch.optim.Adam(self.parameters(), lr=args['lr'], weight_decay=args['wd'])


    def train(self):
        log_mm_detach = self.log_mm.detach()
        Sigma_f_u_detach = self.Sigma_f_u.detach()
        f_u_detach = self.f_u.detach()
        for idx in range(self.args['local_epoch']):
            # re-calculate the values related to the hyper-parameters in the covariance function
            self.recalculate_cov_related_value(Sigma_f_u_detach, f_u_detach)

            # update the parameters of the posterior distribution for the Polya-Gamma random variables
            self.calculate_PG_expectation_f()

            # update the values needed for the posterior of the latent function f on inducing points such as f_u
            self.calculate_posterior_GP_f(log_mm_detach)

            # update the scaling parameter of the intensity function for the cox process i.e. the scaling parameter of the logistic transformation function
            self.update_mm(log_mm_detach)

            # # calculate the Evidence Lower Bound (ELBO)
            # elbo_seq.append(pp_m.calculate_lower_bound())

            negative_elbo = self.calculate_lower_bound()
            # elbo_seq.append(-loss.detach())
            # print(negative_elbo, self.DPM_Kernel.layer1.W_mu.mean())
            # print(negative_elbo, self.DPM_Kernel.layer1.W_rho.mean())
            self.optim.zero_grad()
            negative_elbo.backward(retain_graph=True)
            self.optim.step()
            # print(negative_elbo, self.DPM_Kernel.layer1.W_mu.mean())
            # print(negative_elbo, self.DPM_Kernel.layer1.W_rho.mean())

            log_mm_detach = self.log_mm.detach() 
            Sigma_f_u_detach = self.Sigma_f_u.detach()
            f_u_detach = self.f_u.detach()

    def forward(self):
        pass

    def deep_cov_func_kernel(self, x, x_prime, is_xx=False):
        if (self.DPM_Kernel != None):
            x = self.DPM_Kernel(x.unsqueeze(-1)).squeeze()
            if (is_xx == False):
                x_prime = self.DPM_Kernel(x_prime.unsqueeze(-1)).squeeze()
        if (is_xx):
            return cov_func(x, x, self.cov_params())
        else:
            return cov_func(x, x_prime, self.cov_params())

    def recalculate_cov_related_value(self, Sigma_f_u_detach, f_u_detach):

        # re-calculate the covariance related values
        # self.Kzz = self.deep_cov_func_kernel(self.induced_location, self.induced_location, self.cov_params())+ self.noise*torch.eye(self.num_inducing_points)
        self.Kzz = self.deep_cov_func_kernel(self.induced_location, self.induced_location, is_xx=True) + self.noise*torch.eye(self.num_inducing_points)

        # # L = torch.linalg.cholesky(self.Kzz + self.noise*torch.eye(self.Kzz.shape[0]))
        # L = torch.linalg.cholesky(self.Kzz + self.noise*torch.eye(self.num_inducing_points))
        #
        # # L_inv = solve_triangular(L, torch.eye(L.shape[0]), lower = True, check_finite=False)
        # L_inv = torch.linalg.solve_triangular(L, torch.eye(L.shape[0]), upper = False)
        #
        # self.Kzz_inv = torch.matmul(L_inv.T, L_inv)
        # print(self.Kzz.mean(), self.Kzz.min(), self.Kzz.max())
        # try:
        # print("self.Kzz", self.Kzz, self.Kzz.mean())
        L = torch.linalg.cholesky(self.Kzz)
        self.Kzz_inv = torch.cholesky_inverse(L)
        # except:
        # print(self.Kzz.mean(), self.Kzz.min(), self.Kzz.max())

        self.logdet_Kzz = 2. * torch.sum(torch.log1p(L.diagonal()))

        ##############################################################################
        ##############################################################################


        # self.k_u_X = [self.deep_cov_func_kernel(self.induced_location, data_i, self.cov_params()) for data_i in self.data]
        # self.k_u_X = self.deep_cov_func_kernel(self.induced_location, self.DPM_Data(self.data.unsqueeze(1)).squeeze(), self.cov_params())
        self.k_u_X = self.deep_cov_func_kernel(self.induced_location, self.data)
        # self.kappa_X = [torch.matmul(self.Kzz_inv, k_u_X_i) for k_u_X_i in self.k_u_X]
        self.kappa_X = torch.matmul(self.Kzz_inv, self.k_u_X)


        # self.k_u_int_points = self.deep_cov_func_kernel(self.induced_location,self.integration_location, self.cov_params())
        self.k_u_int_points = self.deep_cov_func_kernel(self.induced_location, self.integration_location)
        self.kappa_int_points = torch.matmul(self.Kzz_inv, self.k_u_int_points)

        self.f_X, self.f2_X = self.predictive_posterior_GP_f(self.data, 'X', current_Sigma_f_u= Sigma_f_u_detach, current_f_u=f_u_detach)
        self.f2_X = self.f2_X + self.f_X**2

        # self.f_X, var_f_X = self.predictive_posterior_GP_f(self.data, 'X', Sigma_f_u_detach, f_u_detach)
        # self.f2_X = var_f_X + self.f_X**2

        self.f_int_points, self.f2_int_points = self.predictive_posterior_GP_f(self.integration_location, 'int_points', current_Sigma_f_u=Sigma_f_u_detach, current_f_u=f_u_detach)
        self.f2_int_points = self.f2_int_points + self.f_int_points**2

        # self.c_X_f = torch.sqrt(self.f2_X)
        # self.c_int_points_f = torch.sqrt(self.f2_int_points)
        #
        # self.w_X_f = 0.5*torch.tanh(0.5*self.c_X_f)/self.c_X_f
        # self.w_int_points_f = 0.5*torch.tanh(0.5*self.c_int_points_f)/self.c_int_points_f
        #
        # self.Lambda_f_without_mm = 0.5*torch.exp(-0.5*self.f_int_points)/torch.cosh(0.5*self.c_int_points_f)

    def place_inducing_points(self):
        """ Places the induced points for sparse GP.
        """

        num_per_dim = torch.ceil(self.num_inducing_points/1)
        # print("num_per_dim", num_per_dim)
        # dist_between_points = (self.data_range[1] - self.data_range[0]) / num_per_dim
        # print("data_range", self.data_range[0], self.data_range[1], dist_between_points.numpy())
        # induced_grid = torch.arange(start = float(self.data_range[0]), end = float(self.data_range[1]), step = float(dist_between_points.numpy()))
        # print("float(dist_between_points.numpy())", float(dist_between_points.numpy()))
        # print("self.induced_grid.shape", induced_grid.shape)
        # induced_grid = torch.arange(start = -3.,end = 3.,step = dist_between_points.numpy()[0])

        induced_grid = torch.linspace(float(self.data_range[0]), float(self.data_range[1]), self.num_integration_points)
        return induced_grid[:self.num_inducing_points]
        # print("self.induced_location.shape", self.induced_location.shape)

    def place_integration_points(self):
        """ Places the integration points for Monte Carlo integration and
        updates all related kernels.
        """

        # self.integration_location = torch.rand(self.num_integration_points)
        # self.integration_location *= self.T
        #
        # max_terminal_point = torch.max(torch.stack([self.data[-1]]))
        # self.integration_location = torch.concat((self.integration_location, torch.tensor([0.5*(self.T-max_terminal_point)])))
        # self.integration_location, _ = torch.sort(self.integration_location)
        self.integration_bin_edges = torch.linspace(self.data_range[0], self.data_range[1], self.num_integration_points+1)

        # bin center point
        self.integration_location = (self.integration_bin_edges[:-1] + self.integration_bin_edges[1:])/2

        # self.num_integration_points += 1
        #
        # induced_grid = torch.arange(start = -3.,end = 3.,step = dist_between_points.numpy()[0])

        # self.integration_location = torch.linspace(-3., 3., self.num_integration_points)

    def calculate_PG_expectation_f(self):
        # corresponding to optimal Polya-Gamma density and optimal Poisson process
        # refer to Eq. (16) and Eq. (19),(20), (21), (22)

        # refer to c_{c,m}, below Eq. (21)
        self.c_X_f = torch.sqrt(self.f2_X)

        # refer to c_c(t), below Eq. (22)
        self.c_int_points_f = torch.sqrt(self.f2_int_points) 

        # refer to Eq. (28)
        self.w_X_f = 0.5*torch.tanh(0.5*self.c_X_f)/self.c_X_f

        # refer to first part of Eq. (30)
        self.w_int_points_f = 0.5*torch.tanh(0.5*self.c_int_points_f)/self.c_int_points_f 

        # refer to part of Eq. (22)
        self.Lambda_f_without_mm = 0.5*torch.exp(-0.5*self.f_int_points)/torch.cosh(0.5*self.c_int_points_f) 

    def calculate_posterior_GP_f(self, log_mm_detach):
        # refer to Optimal Gaussian Process f_c(t), Eq. (24), (25), (26), (27), (31), (32), (33)

        # a_val = (torch.sum(torch.exp(self.log_pi_k), axis=0)**2)-torch.sum(torch.exp(self.log_pi_k*2), axis=0)

        # part of second part of Eq. (26), (27)
        # size: (number of integration points, ) e.g. (1001,)
        Lambda_f = self.Lambda_f_without_mm*torch.exp(log_mm_detach) 

        # second part of Eq. (26)
        # size: (number of integration points, ) e.g. (1001,)
        A_int_points = self.w_int_points_f * Lambda_f

        # second part of Eq. (27)
        # size: (number of integration points, ) e.g. (1001,)
        B_int_points = -0.5*Lambda_f

        # A_X = (self.w_X_f)
        # B_X = self.relations.shape[0]*0.5

        # first part of Eq. (26)
        # size: (1,)
        # A_X = [torch.sum(w_X_f_i)[None] for w_X_f_i in self.w_X_f]
        A_X = self.w_X_f

        # first part of Eq. (27)
        # size: (number of events, 1) e.g. (550,)
        B_X = torch.ones(self.NumE)*0.5


        ##############################################################################
        ##############################################################################

        # first part in Eq. (32) (excluding K_{zz}^{-1} and without inverse operator)
        # x: t_m in Eq. (26)
        # size: (number of inducing points, number of inducing points) e.g. (50, 50)
        kAk = torch.matmul(self.kappa_X, A_X[:, None]*self.kappa_X.T) + \
            torch.matmul(self.kappa_int_points, A_int_points[:, None]*self.kappa_int_points.T) \
            *self.T/self.num_integration_points

        # DPM_edges = self.DPM_Data(self.integration_bin_edges.unsqueeze(1)).squeeze()
        # DPM_interval_lengths = torch.abs(DPM_edges[1:] - DPM_edges[:-1])
        # kAk = torch.matmul(self.kappa_X, A_X[:, None]*self.kappa_X.T) + \
        #     torch.matmul((self.kappa_int_points*(DPM_interval_lengths[None, :])), A_int_points[:, None]*self.kappa_int_points.T)


        # size: (number of inducing points, number of inducing points) e.g. (50, 50)

        # self.Sigma_f_u_inv = torch.sum(torch.stack(kAk), axis=0)+self.Kzz_inv
        self.Sigma_f_u_inv = kAk + self.Kzz_inv + self.noise*torch.eye(self.Kzz_inv.shape[0])
        # L_inv = [torch.linalg.cholesky(self.Sigma_f_u_inv[domain_i]) for domain_i in range(self.numDomains)]

        try:
            L_inverse = torch.linalg.cholesky(self.Sigma_f_u_inv)
        except:
            a = 1

        # Eq. (32)
        self.Sigma_f_u = torch.cholesky_inverse(L_inverse) 


        # # size: ()
        self.logdet_Sigma_f_u = -2*torch.sum(torch.log1p(L_inverse.diagonal())) 

        # Kb: only the integrator term in Eq. (33): \int B(t)K_{z,t}dt
        # size: (number of inducing points, ) e.g. (50,)

        # using client specific mean-functions
        B_X_with_mean = B_X + A_X * (torch.matmul(self.mean_functions()*torch.ones(self.kappa_X.shape[0]), self.kappa_X)-self.mean_functions())
        B_int_points_with_mean = B_int_points + A_int_points * (torch.matmul(self.mean_functions()*torch.ones(self.kappa_int_points.shape[0]), self.kappa_int_points)-self.mean_functions())
        Kb = torch.matmul(self.k_u_X, B_X_with_mean)+(torch.matmul(self.k_u_int_points, B_int_points_with_mean)/self.num_integration_points*self.T)

        # Kb = torch.matmul(self.k_u_X, B_X)+(torch.matmul(self.k_u_int_points, B_int_points)/self.num_integration_points*self.T)
        # Kb = torch.matmul(self.k_u_X, B_X)+(torch.matmul(self.k_u_int_points, DPM_interval_lengths*B_int_points))

        # size: (number of inducing points, ) e.g. (50,)
        self.f_u = torch.matmul(self.Sigma_f_u, torch.matmul(Kb, self.Kzz_inv))

    def gp_prediction(self, inducing_location, origin_mu, test_points):

        K_ori = self.deep_cov_func_kernel(inducing_location, inducing_location)+ self.noise * torch.eye(inducing_location.shape[0])
        L = torch.linalg.cholesky(K_ori )
        # L_inv = torch.linalg.solve_triangular(L, torch.eye(L.shape[0]), upper=False)
        #
        # K_ori_inv = torch.matmul(L_inv.T, L_inv)
        K_ori_inv = torch.cholesky_inverse(L)
        ku_x_prime = self.deep_cov_func_kernel(inducing_location, test_points)
        kappa = torch.matmul(K_ori_inv, ku_x_prime)

        # add mean functions to the prediction
        mu_test = torch.matmul(kappa.T, origin_mu-self.mean_functions())+self.mean_functions()

        return mu_test

    def predictive_posterior_GP_f(self, x_prime, points=None, current_Sigma_f_u = None, current_f_u = None):
        if (current_Sigma_f_u == None):
            current_Sigma_f_u=self.Sigma_f_u
        if (current_f_u == None):
            current_f_u=self.f_u

        if points is None:
            ku_x_prime = self.deep_cov_func_kernel(self.induced_location, x_prime) 
            kappa = torch.matmul(self.Kzz_inv, ku_x_prime)
        elif points == 'int_points':
            ku_x_prime = self.k_u_int_points
            kappa = self.kappa_int_points
        elif points == 'X':
            ku_x_prime = self.k_u_X
            kappa = self.kappa_X

        current_mean_function_value = self.mean_functions()

        mu_f_x_prime = torch.matmul(kappa.T, current_f_u-current_mean_function_value) + current_mean_function_value
        K_xx = self.cov_params()[0]*torch.ones(x_prime.shape[0])
        # K_xx = torch.diag(self.deep_cov_func_kernel(x_prime, x_prime, self.cov_params()))
        var_f_x_prime = K_xx - torch.sum(kappa*(ku_x_prime),axis=0)
        var_f_x_prime[var_f_x_prime<self.noise] = self.noise
        var_f_x_prime += torch.sum(kappa*(torch.matmul(kappa.T, current_Sigma_f_u).T),axis=0)
        # var_f_x_prime = K_xx - torch.sum(kappa*(ku_x_prime - torch.matmul(kappa.T, current_Sigma_f_u).T),axis=0)
        # print('number of negative term is: ', torch.sum(var_f_x_prime<0).numpy())

        return mu_f_x_prime, var_f_x_prime

    def update_mm(self, log_mm_detach):
        # Eq: 23

        # the log pdf of gamma distribution is:
        # log(gamma(x|a,b)) = (a-1)*log(x) - b*x - log(gamma(a)) + a*log(b)
        # where gamma(a) is the gamma function


        # size: (number of integration points, ) e.g. (1001,)
        self.Lambda_f = self.Lambda_f_without_mm * torch.exp(log_mm_detach) 

        # parameters of the posterior distribution of self.mm
        # a_{c,q} in Eq. (23)
        self.alpha_mm_q = torch.sum(self.Lambda_f)/self.num_integration_points*self.T + self.NumE  + self.alpha_mm_prior 
        # b_{c,q} in Eq. (23)
        self.beta_mm_q = self.T + self.beta_mm_prior

        # the posterior of mm is gamma distribution with parameters alpha_mm_q and beta_mm_q
        # the posterior mean of log(mm) is digamma(alpha_mm_q)-log(beta_mm_q)
        # the posterior mean of mm is alpha_mm_q/beta_mm_q
        self.log_mm = torch.digamma(self.alpha_mm_q)-torch.log1p(self.beta_mm_q) 
        self.mm = self.alpha_mm_q/self.beta_mm_q


    def test_criteria(self, time_period, data):
        """ Calculates the log-likelihood, MAE for the specified time period and its corresponding data.

        :return: float
            log-likelihood, MAE
        """
        # a_val = (torch.sum(torch.exp(self.log_pi_k), axis=0)**2)-torch.sum(torch.exp(self.log_pi_k*2), axis=0)
        num_integration_points = 1000
        eval_integration_points = torch.linspace(time_period[0], time_period[1], steps = num_integration_points)
        # print(eval_integration_points, time_period[0], time_period[1])
        eval_integration_functions, _ = self.predictive_posterior_GP_f(eval_integration_points)

        data = torch.tensor(data)
        eval_data_functions, _ = self.predictive_posterior_GP_f(data)

        log_likelihood_1 = self.mm*torch.sum(sigma_f(eval_integration_functions)*((time_period[1]-time_period[0])/num_integration_points))
        log_likelihood_2 = torch.sum(torch.log1p(self.mm*sigma_f(eval_data_functions)))
        MAE = torch.abs(len(data)-log_likelihood_2)

        return - log_likelihood_1 + log_likelihood_2, MAE, len(data)

    def calculate_lower_bound(self):
        """ Calculates the variational lower bound for current posterior.

        :return: float
            Variational lower bound.
        """

        # calculate the lower bound for the first term
        h_int_points_for_f = -self.f_int_points/2-self.f2_int_points*self.w_int_points_f/2-torch.log(torch.tensor(2.0)) 
        integrand_for_f = h_int_points_for_f - torch.log1p(self.Lambda_f_without_mm) - logcosh(self.c_int_points_f/2)  + 0.5 *(self.c_int_points_f**2)*self.w_int_points_f+1 
        LL_f0 = torch.sum(torch.matmul(integrand_for_f, self.Lambda_f_without_mm*torch.exp(self.log_mm))/self.num_integration_points*self.T) 

        # LL_f = LL_f0
        # LL_f = torch.sum(torch.stack(LL_f0))

        h_X_for_f = 0.5*self.f_X-0.5*(self.f2_X)*(self.w_X_f)-torch.log1p(torch.tensor(2.))+self.log_mm 
        summand_for_f = torch.sum(h_X_for_f -logcosh((0.5*self.c_X_f)) +0.5*((self.c_X_f**2))*self.w_X_f) 
        # LL_f += summand_for_f
        # LL_f += torch.sum(torch.stack(summand_for_f))

        # LL_f -= torch.sum(torch.stack([self.numDomains*(self.T*self.mm[domain_i])[0] for domain_i in range(self.numDomains)]))
        # LL_f_term_3 = -(self.T * self.mm)[0]
        LL_f_term_3 = -(self.T * self.mm)


        Sigma_s_mugmug_f = self.Sigma_f_u + torch.outer(self.f_u-self.mean_functions(), self.f_u-self.mean_functions()) 

        # L_f = torch.sum(torch.stack([torch.sum(-.5*torch.trace(torch.matmul(self.Kzz_inv, Sigma_s_mugmug_f[domain_i]))) for domain_i in range(self.numDomains)]))
        L_f_term_1 = torch.sum(-.5*torch.trace(torch.matmul(self.Kzz_inv, Sigma_s_mugmug_f)))

        L_f_term_2 = -torch.sum(.5*self.logdet_Kzz)

        L_f_term_3 = torch.sum(.5*self.logdet_Sigma_f_u + .5*self.num_inducing_points) 

        elbo = LL_f0 + summand_for_f + LL_f_term_3 + L_f_term_1 + L_f_term_2 + L_f_term_3
        # print("elbo1", elbo, LL_f0, summand_for_f, LL_f_term_3, L_f_term_1, L_f_term_2, L_f_term_3)
        # print("elbo1", elbo)
        # ---------
        if (self.args['is_bayesian_paramter'] == True and len(self.global_param_dict) != 0 and len(self.global_param_dict)!=0):
            keys_list = list(self.global_param_dict.keys())
            # print(keys_list)
            client_dist_list = self.DPM_Kernel.dist() + self.mean_functions.dist() + self.cov_params.dist()
            for i in range(0, len(keys_list), 2):
                key1, key2 = keys_list[i], keys_list[i+1]
                klvalue = kl_divergence(client_dist_list[int(i/2)],
                getNormalDist(self.global_param_dict[key1], self.global_param_dict[key2])).mean()
                elbo -= klvalue 
                # print(keys_list[i], keys_list[i+1], klvalue)
            # print()
            # print("elbo2", elbo)
        # print("elbo3", elbo)
        # if (torch.isnan(elbo).any() or torch.isinf(elbo).any()):
        #     print("elbo3", elbo)
        return elbo

