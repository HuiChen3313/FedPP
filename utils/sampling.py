import torch
from utils.utils import gp_prediction
from scipy.stats import poisson, uniform, gamma, dirichlet, norm
from FLAlgorithms.trainmodel.DLMKernel import OneLayerMLP

def sample_inhomogeneous_poisson_process(maximum_val, T, start_point, function_type, inducing_location, inducing_mu, cov_params_k,  delta_val = 1.):

    expected_nums = ((T-start_point) * maximum_val)
    actual_num_1 = poisson.rvs(mu=expected_nums)

    if actual_num_1>0:
        candidate_1, _ = torch.sort(torch.rand(size=[actual_num_1]) * (T-start_point)+start_point)
        if function_type == 'GP':
            # try:
            f_1 = gp_prediction(inducing_location, inducing_mu, cov_params_k, candidate_1)
            # except:
            #     a = 1
            sigma_f_k = 1 / (1 + torch.exp(-f_1))
        elif function_type == 'Expo':
            sigma_f_k = torch.exp(-delta_val*(candidate_1-start_point))

        time_points, _ = torch.sort(candidate_1[torch.rand(size=[actual_num_1]) < sigma_f_k])
    else:
        time_points = []

    return time_points

def inhomogeneous_poisson_process_sample(sampling_range, num_clients = 2):

    function_type = 'GP'
    start_point = sampling_range[0]
    T = sampling_range[1]
    m_val = torch.tensor([50.0])
    inducing_location = torch.linspace(start_point, T, 50)
    inducing_U = torch.normal(mean = torch.zeros_like(inducing_location), std = torch.ones_like(inducing_location))
    cov_params_f_candidate_list = [[1.5, 10], [2.0, 8], [2, 7], 
                                   [1, 3.5], [2, 2.2]]
    cov_params_f = torch.tensor([cov_params_f_candidate_list[i%len(cov_params_f_candidate_list)] for i in range(num_clients)])

    data = []
    for i in range(num_clients):
        data.append(sample_inhomogeneous_poisson_process(m_val, T, start_point, function_type,
                                                         inducing_location, inducing_U,  cov_params_f[i]))

    return data, cov_params_f, [m_val for _ in range(num_clients)], [inducing_location for _ in range(num_clients)], [inducing_U for _ in range(num_clients)] 