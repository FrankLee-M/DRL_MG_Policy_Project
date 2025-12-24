from ast import Not
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from my_project.RA_obs.config_obs import*
import os
# 检查每天的电价数组是否为 24 组
# 读取CSV文件
# res,lambda_hv prediciton data : array
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_file_path():
    # 获取当前文件夹路径
    current_file_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_file_path)
    # '/Users/xinli/Documents/PyCharmProject/BLO-DRL project/my_project/environment/observations/'

    # 存储所有 CSV 文件的路径
    csv_files = []

    # 遍历当前文件夹
    for file_name in os.listdir(current_folder):
        # 检查文件是否以 .csv 结尾
        if file_name.endswith(".csv"):
            # 获取文件的完整路径
            file_path = os.path.join(current_folder, file_name)
            csv_files.append(file_path)

    # # 打印结果
    # print("CSV 文件路径列表:")
    # for file_path in csv_files:
    #     print(file_path)
    csv_files.sort()
    return csv_files


class DataProcess():
    """
    functions:
    ----------
    get_PJM_data/get_PV_data :

    """

    def __init__(self, range_res=range_res, range_lmhv=range_lmhv) -> None:
        """
        Parameters:
        range_res/range_res: scale the data
        
        functions:
        get_period_data: get data during start-to-end date
        get_PJM_data/get_PV_data: 
        """
        self.range_res = range_res
        self.range_lmhv = range_lmhv
    
    
    def read_data(self, file):
        df = pd.read_csv(file)
        if os.path.basename(file) == "PJM-HourlyRealTime.csv":
            
            df['HOURBEGINNING_TIME'] = pd.to_datetime(
                df['HOURBEGINNING_TIME'], format='%d-%b-%Y %H:%M:%S')
        else:
            if os.path.basename(file) == "ninja_pv.csv":
                df['HOURBEGINNING_TIME'] = pd.to_datetime(
                df['HOURBEGINNING_TIME'], format='%Y-%m-%d %H:%M')
             

        df['date'] = df['HOURBEGINNING_TIME'].dt.date
        df['hour'] = df['HOURBEGINNING_TIME'].dt.hour
        
        # check 按日期分组并计数每组的记录数
        daily_counts = df.groupby('date').size()
        problematic_days = daily_counts[daily_counts != 24]
        if not problematic_days.empty:  # pyright: ignore[reportAttributeAccessIssue]
            print('problematic_days',problematic_days)
        
        return df
            
        
    def get_period_data(self, filename , start_date, end_date):
        df = self.read_data(filename)
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()  # Convert to datetime.date
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        monthly_data = df.loc[mask]
        return monthly_data

    def get_PJM_data(self, data):

        data['RECO'] = data['RECO'].replace(
            '[\$,]', '', regex=True).astype(float) #type:ignore

        price_data = self.range_lmhv*(data['RECO']._values)
        price_data = price_data[::-1]  # 修正时间顺序
        return price_data

    def get_PV_data(self, data):

        data['electricity'] = data['electricity'].astype(float)

        electricity_data = self.range_res*(data['electricity']._values)
        return electricity_data


DP = DataProcess()
csv_files = get_file_path()
filename = csv_files[0]
start_date,end_date = "2022-01-01 00:00","2022-12-10 23:00"
monthly_data = DP.get_period_data(filename,start_date,end_date)
lambda_hv_data = DP.get_PJM_data(monthly_data)
# plt.plot(lambda_hv_data)
# plt.show()
lambda_hv_data = np.tile(lambda_hv_data,data_loop_num)
# print('hv_price_length:',lambda_hv_data.size)


filename = csv_files[1]
monthly_data = DP.get_period_data(filename,start_date,end_date)
res_data = DP.get_PV_data(monthly_data)
# plt.plot(res_data)
# plt.show()
res_data = np.tile(res_data,data_loop_num)
# print('res_length:', res_data.size)

