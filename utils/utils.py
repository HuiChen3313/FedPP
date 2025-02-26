from scipy.stats import poisson, uniform, gamma, dirichlet, norm
# from scipy.linalg import solve_triangular
# from scipy.special import digamma, logsumexp, gammaln
import time
import copy, random
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
from scipy.integrate import quadrature
import torch.nn.functional as F
from sys import getsizeof, stderr
from itertools import chain
from collections import deque

def exp_with_upperbound(parameter):
    threshold = torch.tensor(88.0)
    parameter = torch.where(parameter < threshold, 
                            torch.expm1(parameter), 
                            torch.expm1(threshold))
    return parameter

def logcosh(x):
    # s always has real part >= 0
    s = torch.sign(x) * x
    p = torch.exp(-2 * s)
    return s + torch.log1p(p) - torch.log(torch.tensor(2.))

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

def sample_gp(x, cov_params):
    num_points = len(x)
    K = cov_func(x, x, cov_params)
    L = torch.linalg.cholesky(K+1e-8 * torch.eye(num_points))
    rand_nums = torch.random.randn(num_points)

    return torch.dot(L, rand_nums)

# def compare_process(contec):
#     # print(contec.origin_m_val)
#     # print(torch.exp(contec.log_m))
#     # fig, ax = plt.subplots()
#     plt.figure(4)
#     test_points = torch.linspace(0, contec.T, 2000)
#     f_ori = []
#     f_pre = []
#     g_ori = []
#     g_pre = []
#     for kk in range(contec.K):
#         plt.subplot(contec.K+1, 1, 1+kk)
#         origin_val, origin_variance = gp_prediction_for_plot(contec.origin_points, contec.origin_mu_f[kk], contec.origin_cov_f[kk], test_points)
#         predict_val, predict_variance = gp_prediction_for_plot(contec.induced_points, contec.f_u[kk], contec.cov_params[kk], test_points)
#         # predict_val, predict_variance = gp_prediction_for_plot(contec.induced_points, norm.rvs(size=len(contec.f_u[kk])), contec.cov_params[kk], test_points)
#         f_ori.append(origin_val)
#         f_pre.append(predict_val)
#         plt.plot(test_points, origin_val, label = 'Origin')
#         plt.plot(test_points, predict_val, label = 'SCGP fitted')
#         plt.fill_between(test_points, predict_val+predict_variance**(0.5), predict_val-predict_variance**(0.5), color='yellow', alpha = 0.5)
#         # plt.plot([data, data], [-1, 1], c = 'red')
#         plt.legend()
#         plt.title('F val')
#
#     predict_pi = torch.sum(contec.pi_k, axis=0)**2-torch.sum(contec.pi_k**2, axis=0)
#     origin_pi = torch.sum(contec.origin_pis, axis=0)**2-torch.sum(contec.origin_pis**2, axis=0)
#     plt.subplot(contec.K+1, 1, 1+(contec.K))
#     plt.plot(test_points, (1/(1+torch.exp(-torch.asarray(f_ori).T))).dot(origin_pi), label = 'Origin')
#     plt.plot(test_points, (1/(1+torch.exp(-torch.asarray(f_pre).T))).dot(predict_pi), label = 'SCGP fitted')
#     # plt.plot([data, data], [-1, 1], c = 'red')
#     # plt.legend()
#     plt.title('F val')
#     plt.savefig('F function later vis ' + str(contec.alpha_adam) + '.pdf', edgecolor='none', format='pdf', bbox_inches='tight')
#     plt.show()
#
#     # plt.figure(2)
#     # for kk in range(contec.K):
#     #     plt.subplot(contec.K, 2, 1+2*kk)
#     #     origin_val = gp_prediction(contec.origin_points, contec.origin_mu_g[kk], contec.origin_cov_g[kk], test_points)
#     #     predict_val = gp_prediction(contec.induced_points, contec.g_s[kk], contec.cov_params_g[kk], test_points)
#     #     g_ori.append(origin_val)
#     #     g_pre.append(predict_val)
#     #     plt.plot(test_points, origin_val, label = 'Origin')
#     #     plt.plot(test_points, predict_val, label = 'SCGP fitted')
#     #     if kk ==0:
#     #         plt.legend()
#     #     plt.title('G val')
#     #
#     #     plt.subplot(contec.K, 2, 2+2*kk)
#     #     plt.plot(test_points, contec.origin_m_val[kk]*(1/(1+torch.exp(-torch.asarray(origin_val)))), label = 'Origin')
#     #     plt.plot(test_points, contec.m_k[kk]*(1/(1+torch.exp(-torch.asarray(predict_val)))), label = 'SCGP fitted')
#     #
#     #     plt.title('Sigmoid Cox')
#     #
#     #
#     # # plt.subplot(contec.K+1, 1, 1+(contec.K))
#     # # plt.plot(test_points, (1/(1+torch.exp(-torch.asarray(g_ori).T))).dot(contec.origin_m_val), label = 'Origin')
#     # # plt.plot(test_points, (1/(1+torch.exp(-torch.asarray(g_pre).T))).dot(torch.exp(contec.log_m)), label = 'SCGP fitted')
#     # # # plt.legend()
#     # # plt.title('G val')
#     #
#     #
#     # plt.show()

def sigma_f(val_points):
    return (1+torch.exp(-val_points))**(-1)


def cov_func(x, x_prime, cov_params):
    """ Computes the covariance functions between x and x_prime.

    :param x: torch.ndarray [num_points x D]
        Contains coordinates for points of x
    :param x_prime: torch.ndarray [num_points_prime x D]
        Contains coordinates for points of x_prime
    :param cov_params: list
        First entry is the amplitude, second an D-dimensional array with
        length scales.

    :return: torch.ndarray [num_points x num_points_prime]
        Kernel matrix.
    """

    theta_1, theta_2 = F.softplus(cov_params[0]), F.softplus(cov_params[1])

    x_theta2 = x[:, None] / theta_2
    xprime_theta2 = x_prime[None,:] / theta_2
    h = x_theta2** 2 - 2. * x_theta2*xprime_theta2 + xprime_theta2** 2
    return theta_1 * torch.exp(-.5*h)

# def gp_prediction_for_plot(origin_points, origin_mu, cov_params_k, test_points):
#     noise = 0.01
#     K_ori = cov_func(origin_points, origin_points, cov_params_k)
#     L = torch.linalg.cholesky(K_ori + noise * torch.eye(K_ori.shape[0]))
#     L_inv = torch.linalg.solve_triangular(L, torch.eye(L.shape[0]), upper=False)
#
#     K_ori_inv = L_inv.T.dot(L_inv)
#
#     ku_x_prime = cov_func(origin_points, test_points, cov_params_k)
#     kappa = K_ori_inv.dot(ku_x_prime)
#
#     mu_test = kappa.T.dot(origin_mu)
#
#     K_xx = cov_params_k[0] * torch.ones(len(test_points))
#     var_f_x_prime = K_xx - torch.sum(kappa * (ku_x_prime - kappa.T.dot(K_ori).T), axis=0)
#
#     return mu_test, var_f_x_prime


def gp_prediction(inducing_location, origin_mu, cov_params_k, test_points):
    noise = 0.0001
    K_ori = cov_func(inducing_location, inducing_location, cov_params_k)+ noise * torch.eye(inducing_location.shape[0])
    L = torch.linalg.cholesky(K_ori )
    # L_inv = torch.linalg.solve_triangular(L, torch.eye(L.shape[0]), upper=False)
    #
    # K_ori_inv = torch.matmul(L_inv.T, L_inv)

    K_ori_inv = torch.cholesky_inverse(L)

    ku_x_prime = cov_func(inducing_location, test_points, cov_params_k)
    kappa = torch.matmul(K_ori_inv, ku_x_prime)

    mu_test = torch.matmul(kappa.T, origin_mu)

    return mu_test

def split_1d_tensor_XSplit(tensor, num_splits):
    tensor = torch.tensor(tensor)
    shuffled_tensor = tensor[torch.randperm(len(tensor))]

    split_size = len(tensor) // num_splits
    remainder = len(tensor) % num_splits
    split_tensors = []

    start_idx = 0
    for i in range(num_splits):
        end_idx = start_idx + split_size + (1 if i < remainder else 0)
        split_tensors.append(shuffled_tensor[start_idx:end_idx])
        start_idx = end_idx

    for split in split_tensors:
        split, _ = split.sort()

    return split_tensors

def split_1d_tensor(tensor, split_ratio = 0.2): 

    shuffled_tensor = tensor[torch.randperm(tensor.size(0))]

    split_idx = int(0.2 * len(shuffled_tensor))
    split_idx_2 = int(0.4 * len(shuffled_tensor))

    tensor_20_1 = shuffled_tensor[:split_idx]
    tensor_20_2 = shuffled_tensor[split_idx:split_idx_2]
    tensor_60 = shuffled_tensor[split_idx_2:]
    tensor_20_1, _ = tensor_20_1.sort()
    tensor_20_2, _ = tensor_20_2.sort()
    tensor_60, _ = tensor_60.sort()
    
    return tensor_20_1, tensor_20_2, tensor_60

def split_1d_tensor_first_ratio(tensor, split_ratio = 0.2): 
    if (split_ratio >= 1):
        return tensor

    shuffled_tensor = tensor[torch.randperm(tensor.size(0))]

    split_idx = int(split_ratio * len(shuffled_tensor))

    tensor_20 = shuffled_tensor[:split_idx]
    tensor_20, _ = tensor_20.sort()

    return tensor_20