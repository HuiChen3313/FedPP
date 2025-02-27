import time, torch
import numpy as np
import torch.nn.functional as F
from FLAlgorithms.trainmodel.BBBLinear import softplus_inverse
from FLAlgorithms.users.userFedNPP import UserFedNPP
import matplotlib.pyplot as plt
from utils.utils import gp_prediction, sigma_f

# Implementation for FedAvg Server
class FedNPP():
    def __init__(self, save_data, args):
        self.args = args
        self.save_data = save_data
        self.users = []
        for idx in range(args['clients_num']):
            self.users.append(UserFedNPP(idx, num_integration_points = 5000, num_inducing_points = 50, train_data=save_data, args=args))
            self.users[-1].induced_location.requires_grad = True
            self.users[-1].mean_functions.requires_grad = False
        self.avg_cov_param = None
        self.param_dict = None

        print('')
        print('##################################################')
        print('')
        print('parameters to be optimized: ')
        print('')
        for name, param in self.users[0].named_parameters():
            if param.requires_grad:
                print(name)
        print('')
        print('##################################################')
        print('')

    def jundge_aggregate_param(self, name, param):
        if (name == "induced_location" or param.requires_grad == False):
            return True
        return False

    def send_parameters(self):
        for name, param in self.users[0].named_parameters():
            if self.jundge_aggregate_param(name, param):
                continue
            param.data = self.param_dict[name].data.clone()

        for i in range(0,self.args['clients_num']):
            self.users[i].global_param_dict = self.param_dict

    def select_users(self, round, num_users):
        if(num_users == len(self.users)):
            print("All users are selected")
            return self.users
        num_users = min(num_users, len(self.users))
        return np.random.choice(self.users, num_users, replace=False) #, p=pk)

    def aggregate_parameter(self):
        self.param_dict = {}
        rho_name_list = []

        for i in range(0, self.args['clients_num']):
            for name, param in self.users[i].named_parameters():
                if self.jundge_aggregate_param(name, param):
                    continue
                if (i == 0):
                    self.param_dict[name] = param.clone().detach()
                    if ('rho' in name):
                        rho_name_list.append(name)
                else:
                    self.param_dict[name] += param.clone().detach()

        for name in self.param_dict.keys():
            self.param_dict[name] /= self.args['clients_num']

        def update_aggerate_parameter():
            if (self.args['aggerate_method'] == 'FedPP'):
                return (softplus_inverse(param).clone().detach())**2 + \
                            (one_client_param_dict[mu_name].clone().detach())**2 - \
                                (self.param_dict[mu_name].clone().detach())**2
            if (self.args['aggerate_method'] == 'AggSigma2'):
                return (softplus_inverse(param).clone().detach())**2

            if (self.args['aggerate_method'] == 'AggSigma'):
                return (softplus_inverse(param).clone().detach())

        # ['FedAvg', 'AggSigma', 'AggSigma2', 'FedPP']
        if (self.args['aggerate_method'] != 'FedAvg'):
            for i in range(0,self.args['clients_num']):
                one_client_param_dict = {}
                for name, param in self.users[i].named_parameters():
                    one_client_param_dict[name] = param
                    if self.jundge_aggregate_param(name, param) or name not in rho_name_list:
                        continue
                    mu_name = name.replace('rho', 'mu')
                    if (i == 0):
                        self.param_dict[name] = update_aggerate_parameter()
                    else:
                        self.param_dict[name] += update_aggerate_parameter()
                     
            for name in self.param_dict.keys():
                if (name in rho_name_list):
                    self.param_dict[name] /= self.args['clients_num']
                    if (self.args['aggerate_method'] == 'AggSigma2' or self.args['aggerate_method'] == 'FedPP'):
                        self.param_dict[name] = torch.sqrt(self.param_dict[name])
                    self.param_dict[name] = softplus_inverse(self.param_dict[name])

    def train(self):
        def test_performance(data):
            # test_range = [100, 150]
            # test_data = []
            test_log_likelihood = []
            test_MAE = []
            test_num_list = []
            for domain_i in range(self.args['clients_num']):
                ll_domain_i, mae_domain_i, data_num_i = self.users[domain_i].test_criteria(self.save_data['sampling_range'], data[domain_i])
                test_log_likelihood.append(ll_domain_i)
                test_MAE.append(mae_domain_i)
                test_num_list.append(data_num_i)
            # print('log likelihood: ', torch.sum(torch.tensor(test_log_likelihood)).detach().numpy() / sum(test_num_list))
            # print('MAE: ',            torch.sum(torch.tensor(test_MAE)).detach().numpy() / sum(test_num_list))
            return torch.sum(torch.tensor(test_log_likelihood)).detach().numpy() / sum(test_num_list), torch.sum(torch.tensor(test_MAE)).detach().numpy() / sum(test_num_list)
        # elbo_seq = []
        # start_time = time.time()
        print('formal start')
        best_valid_ll = -1e9
        test_ll = test_rmse = 0
        for ite in range(int(self.args['max_epoch'])):
            print(ite, int(self.args['max_epoch']))
            if (ite % self.args['select_period'] == 0):
                self.selected_users = self.select_users(ite, int(len(self.users)/2))
            for user in self.selected_users:
                user.train()
                # if (self.param_dict != None):
                    # print(self.param_dict['DPM_Kernel.layer2.bias_mu']) 
            valid_ll, valid_rmse = test_performance(self.save_data['data_valid'])
            if (best_valid_ll < valid_ll):
                best_valid_ll = valid_ll 
                print("best valid ll", valid_ll)
                test_ll, test_rmse = test_performance(self.save_data['data_test'])
                print("test ll", test_ll)
            if (self.args['local_only'] == 0):
                self.aggregate_parameter()
                self.send_parameters()

        print('----------------------')
        print('cov_params: ', self.users[0].cov_params)
        print('m_val: ', self.users[0].mm)
        print('inducing location: ', self.users[0].induced_location)
        print('mean functions: ', self.users[0].mean_functions)
        print('----------------------')
        return test_ll, test_rmse
    
    def plt_result_figure(self):
        plt.figure(figsize=(10, 5))
        model_coor = torch.linspace(self.save_data['sampling_range'][0], self.save_data['sampling_range'][1], 200)
        for domain_i in range(self.args['clients_num']):
            if (self.args['dataset_type'] == "Synthetic"):
                f_1 = gp_prediction(self.save_data['inducing_location_data'][domain_i], self.save_data['inducing_U_data'][domain_i], self.save_data['cov_params_data'][domain_i], model_coor)
                f_1 = f_1.detach()
            else:
                f_1, _ = self.users[domain_i].predictive_posterior_GP_f(model_coor)
                f_1 = f_1.detach()
            plt.subplot(2, self.args['clients_num'], self.args['clients_num'] * 0 + domain_i + 1)
            plt.plot(model_coor, f_1, label='data')
            # plt.plot([self.users[domain_i].DPM_Data(data[domain_i].unsqueeze(1)).squeeze().detach().numpy(), self.users[domain_i].DPM_Data(data[domain_i].unsqueeze(1)).squeeze().detach().numpy()], [-0.1, 0.1], color='black')
            plt.plot([self.save_data['data'][domain_i].detach().numpy(), self.save_data['data'][domain_i].detach().numpy()], [-0.1, 0.1], color='black')
            # f_self.users = self.users[domain_i].gp_prediction(self.users[domain_i].induced_location, self.users[domain_i].f_u, model_coor)
            f_pp_m, _ = self.users[domain_i].predictive_posterior_GP_f(model_coor)
            plt.plot(model_coor, f_pp_m.detach(), label = 'model-'+str(domain_i))
            plt.legend()

            plt.subplot(2, self.args['clients_num'], self.args['clients_num'] * 1 + domain_i + 1)
            if (self.args['dataset_type'] == "Synthetic"):
                plt.plot(model_coor, self.save_data['m_data'][domain_i]*sigma_f(f_1), label = 'data')
            else:
                plt.plot(model_coor, self.users[domain_i].mm.detach()*sigma_f(f_pp_m.detach()), label = 'data')

            # plt.plot([data[domain_i].numpy(), data[domain_i].numpy()], [-0.1, 0.1], color='black')
            plt.plot(model_coor, self.users[domain_i].mm.detach()*sigma_f(f_pp_m.detach()), label = 'FedEvent')
            # plt.plot(model_coor, pp_m.mm.detach()*sigma_f(f_pp_m.detach()), label = 'model')
            plt.legend()

        plt.savefig('result/simulation_PoissonProcess.pdf', edgecolor='none', format='pdf', bbox_inches='tight')
        plt.show()
