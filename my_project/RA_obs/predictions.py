from .data_process import lambda_hv_data,res_data,data_loop_num
import numpy as np
from my_project.RA_obs.config_obs import predict_length,historical_length

# RA observation model

# def get_all_data():
#     return lambda_hv_data,p_res_data

# def get_initial_len():
#     l1 = min(len(lambda_hv_data),len(p_res_data))
#     l0 = int(l1/data_loop_num)
#     return l0,data_loop_num



def HVprice_preds(
    predict_length : int, # the length of obtained data
    t_start : int # prediction start hour index
    ):
    
    """
    计算预测值的函数。

    参数:
    T (int): 预测数据的长度，表示要进行预测的数据点数量。通常是一个整数值。
    t (int): 开始预测的小时时刻。

    返回:
    array data: 返回与输入参数T和t相关的预测结果。
    """
    lambda_hv = lambda_hv_data[t_start:t_start+predict_length]
    return lambda_hv

# 实时预测
def RES_preds(
    predict_length : int, # the length of obtained data
    t_start : int # prediction start hour index
              ):

    """
    计算预测值的函数。

    参数:
    T (int): 预测数据的长度，表示要进行预测的数据点数量。通常是一个整数值。
    t (int): 开始预测的小时时刻。

    返回:
    array data: 返回与输入参数T和t相关的预测结果。
    """

    p_res = res_data[t_start:t_start+predict_length]
    return p_res

def get_observations(t_start):
    res_pre = res_data[t_start:t_start+predict_length]
    res_his = res_data[t_start-historical_length:t_start]
    
    lmhv_pre = lambda_hv_data[t_start:t_start+predict_length]
    lmhv_his = lambda_hv_data[t_start-historical_length:t_start]
    
    return  np.concatenate((res_his, res_pre)),np.concatenate((lmhv_his, lmhv_pre))

    
    

# def Tau_preds(Tmax, T, t):
#     # np.random.seed(0)
#     # tau_list = np.random.normal(0.05, 0.1, (1, Tmax+T))
#     # tau = tau_list[0][t:t+T]
#     tau = np.zeros((1,T))
#     tau = tau[0][:T]

#     return tau


