from utils.utils import split_1d_tensor, split_1d_tensor_XSplit, split_1d_tensor_first_ratio
from utils.sampling import inhomogeneous_poisson_process_sample
import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle

def load_dataset(args):
    data_file_list = ["data_test", "data_valid", "data", "cov_params_data", "m_data", "inducing_location_data", "inducing_U_data"]
    save_data = {"data":[], "cov_params_data":[], "m_data":[], "inducing_location_data":[], "inducing_U_data": []}
    if (args['dataset_type'] == 'Synthetic'):
        sampling_range = [0., 100.]
        if (args['sampling_data']):
            data, cov_params_data, m_data, inducing_location_data, inducing_U_data = inhomogeneous_poisson_process_sample(sampling_range, args['clients_num'])
            save_data = {"data":data, "cov_params_data":cov_params_data, "m_data":m_data, "inducing_location_data":inducing_location_data, "inducing_U_data": inducing_U_data}
            save_data['data_test'] = [0 for client_idx in range(args['clients_num'])]
            save_data['data_valid'] = [0 for client_idx in range(args['clients_num'])]
            for client_idx in range(args['clients_num']):
                save_data['data_test'][client_idx], save_data['data_valid'][client_idx], save_data['data'][client_idx] = split_1d_tensor(save_data['data'][client_idx])

            for name in save_data.keys():
                for client_idx in range(len(save_data['data'])):
                    with open("./data/Synthetic/" + name + str(sampling_range) + "_" + str(client_idx) + ".npy", 'wb') as f:
                        np.save(f, save_data[name][client_idx].detach().numpy())
            
        else:
            save_data = {"data_test": [], "data_valid": [], "data":[], "cov_params_data":[], "m_data":[], "inducing_location_data":[], "inducing_U_data": []}
            if (args['dataset_only_1type'] == 0):
                if (args['mix_type_num'] != 1):
                    save_data["cov_params_data"] = [None for _ in range(args['clients_num'])]
                    save_data["inducing_location_data"] = [None for _ in range(args['clients_num'])]
                    save_data["inducing_U_data"] = [None for _ in range(args['clients_num'])]
                    data_file_list = ["data_test", "data_valid", "data"]
                else: 
                    data_file_list = ["data_test", "data_valid", "data", "cov_params_data", "m_data", "inducing_location_data", "inducing_U_data"]

                    
                for client_idx in range(args['clients_num']):
                    for name in data_file_list:
                        for mix_id in range(args['mix_type_num']):
                            with open("./data/Synthetic/" + name + str(sampling_range) + "_" + str((client_idx * args['mix_type_num'] + mix_id) % 5) + ".npy", 'rb') as f:
                                if (mix_id == 0):
                                    save_data[name].append(torch.tensor(np.load(f)))
                                else:
                                    save_data[name][-1] = torch.cat((save_data[name][-1], torch.tensor(np.load(f))), dim=0)
                    save_data['m_data'].append(len(save_data['data'][-1]))


            else:
                for client_idx in range(args['clients_num']):
                    for name in data_file_list:
                        with open("./data/Synthetic/" + name + str(sampling_range) + "_" + str(args['dataset_only_1type']) + ".npy", 'rb') as f:
                            save_data[name].append(torch.tensor(np.load(f)))
                one_seq_split = split_1d_tensor_XSplit(save_data['data'][0], args['clients_num'])
                save_data['data'] = [one_seq_split[idx] for idx in range(args['clients_num'])]
                one_seq_split = split_1d_tensor_XSplit(save_data['data_test'][0], args['clients_num'])
                save_data['data_test'] = [one_seq_split[idx] for idx in range(args['clients_num'])]
                one_seq_split = split_1d_tensor_XSplit(save_data['data_valid'], args['clients_num'])
                save_data['data_valid'] = [one_seq_split[idx] for idx in range(args['clients_num'])]
                    

        
    start_point_dict = {'taobao':151150, 'amazon':0, 'conttime':0, 'retweet':0, 'stackoverflow':1325}
    type_num = {'taobao':17, 'amazon':16, 'conttime':5, 'retweet':3, 'stackoverflow':10}

    if (args['dataset_type'] in ['taobao', 'amazon', 'conttime', 'retweet', 'stackoverflow']):
        sampling_range = [1e9, -1e9]
        def load_alltype_dataset(dataset_type='train'):
            if (dataset_type == 'train'):
                file_name = 'train'
                index_name = 'data'
            elif (dataset_type == 'valid'):
                file_name = 'dev'
                index_name = 'data_valid'
            else:
                file_name = 'test'
                index_name = 'data_test'
            save_data[index_name] = []
            for client_idx in range(args['clients_num']):
                one_seq = []
                for mix_idx in range(args['mix_type_num']):
                    with open('data/' + args['dataset_type'] + '/type_'+ str((client_idx * args['mix_type_num'] + mix_idx)%type_num[args['dataset_type']] + 1) + '/' + file_name + '.pkl', 'rb') as file:
                        data = pickle.load(file)  
                        # print(data[0])
                        for idx_data in range(len(data)):
                            for idx_seq in range(len(data[idx_data])):
                                if (data[idx_data][idx_seq]['time_since_start'] >= start_point_dict[args['dataset_type']]):
                                    one_seq.append(float(data[idx_data][idx_seq]['time_since_start'] - start_point_dict[args['dataset_type']])) 

                one_seq.sort()
                save_data[index_name].append(torch.tensor(one_seq))
                # print("sampling_range", sampling_range)
                if (dataset_type == 'train'):
                    sampling_range[0] = min(min(one_seq), sampling_range[0])
                    sampling_range[1] = max(max(one_seq), sampling_range[1])
                    save_data['m_data'].append(len(data))
                    # print("m_data", len(data))
                    save_data["cov_params_data"].append(None)
                    save_data["inducing_location_data"].append(None)
                    save_data["inducing_U_data"].append(None)

        def load_onetype_dataset(dataset_type='train', typeId = 0):
            if (dataset_type == 'train'):
                file_name = 'train'
                index_name = 'data'
            elif (dataset_type == 'valid'):
                file_name = 'dev'
                index_name = 'data_valid'
            else:
                file_name = 'test'
                index_name = 'data_test'
            save_data[index_name] = []
            one_seq = []
            with open('data/' + args['dataset_type'] + '/type_'+ str((typeId)%type_num[args['dataset_type']] + 1) + '/' + file_name + '.pkl', 'rb') as file:
                data = pickle.load(file) 
                for idx_data in range(len(data)): 
                    for idx_seq in range(len(data[idx_data])): 
                        if (data[idx_data][idx_seq]['time_since_start'] >= start_point_dict[args['dataset_type']]): 
                            one_seq.append(float(data[idx_data][idx_seq]['time_since_start'] - start_point_dict[args['dataset_type']])) 
            one_seq_split = split_1d_tensor_XSplit(one_seq, args['clients_num'])
            save_data[index_name] = [one_seq_split[idx] for idx in range(args['clients_num'])]
            # print("sampling_range", sampling_range)
            if (dataset_type == 'train'):
                sampling_range[0] = min(min(one_seq), sampling_range[0])
                sampling_range[1] = max(max(one_seq), sampling_range[1])
                print("m_data", len(data))
                save_data['m_data'] = [len(save_data[index_name][idx]) for idx in range(args['clients_num'])]
                save_data["cov_params_data"] = [None for _ in range(args['clients_num'])]
                save_data["inducing_location_data"] = [None for _ in range(args['clients_num'])]
                save_data["inducing_U_data"] = [None for _ in range(args['clients_num'])]
                
        if (args['dataset_only_1type'] == 0):
            load_alltype_dataset(dataset_type='train')
            load_alltype_dataset(dataset_type='test')
            load_alltype_dataset(dataset_type='valid')
        else:
            load_onetype_dataset(dataset_type='train', typeId = args['dataset_only_1type']-1)
            load_onetype_dataset(dataset_type='test', typeId = args['dataset_only_1type']-1)
            load_onetype_dataset(dataset_type='valid', typeId = args['dataset_only_1type']-1)

    save_data['data'] = [save_data['data'][idx].sort()[0] for idx in range(args['clients_num'])] 
    save_data['data_test'] = [save_data['data_test'][idx].sort()[0] for idx in range(args['clients_num'])] 
    save_data['data_valid'] = [save_data['data_valid'][idx].sort()[0] for idx in range(args['clients_num'])] 
    save_data['data'] = [split_1d_tensor_first_ratio(save_data['data'][idx], args['seleted_clent_data_ratio']) for idx in range(len(save_data['data']))]
    save_data['data_test'] = [split_1d_tensor_first_ratio(save_data['data_test'][idx], args['seleted_clent_data_ratio']) for idx in range(len(save_data['data_test']))]
    save_data['data_valid'] = [split_1d_tensor_first_ratio(save_data['data_valid'][idx], args['seleted_clent_data_ratio']) for idx in range(len(save_data['data_valid']))]
    save_data['sampling_range'] = sampling_range
    print('cov_params_data: ', save_data['cov_params_data'])
    print('m_data: ', save_data['m_data'])
    print('number of clients: ', len(save_data['data']))
    return save_data