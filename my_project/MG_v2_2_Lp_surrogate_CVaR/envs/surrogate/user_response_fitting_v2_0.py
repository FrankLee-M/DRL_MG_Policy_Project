import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os,sys  # [新增] 用于处理路径
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from scipy.stats import norm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(parent_dir)
sys.path.append(project_dir)



from envs.env_mg import MgComplaintFBEnv
from my_project.RA_obs.data_process import lambda_hv_data
from envs.config_para_mg_Nd_50 import begin_t,end_t

"""
Version: v2_0
Based on: v1_2

Major Changes:
- [BREAKING] 对数据划分 训练、验证集、测试集 (Cost-based)

"""

### 拟合 投诉成本
# [修改] 定义保存目录
SAVE_DIR = "my_project/RA_obs/user_response_model_v2_0"
# [新增] 确保目录存在
os.makedirs(SAVE_DIR, exist_ok=True)

class UserResponseCollector:
    """
    第一步：数据采集
    """
#ANCHOR - [修改] start_idx/end_idx,与 MG-model-训练保持一致
    def __init__(self, start_month=5, end_month=6):
        self.env = MgComplaintFBEnv(
            is_user_dynamic=False,
            train_mode=False
        )
        self.env.unit_costs *= 0.0001  
        self.start_idx = begin_t
        self.end_idx = end_t
        
    def collect_data(self):
        print(f"Start collecting data from hour {self.start_idx} to {self.end_idx}...")
        
        data_list = []
        self.env.reset()
        
        # 重新初始化用户状态
        hour_idx_0 = self.start_idx % 24
        self.env.User._random_init_user_model(np.random.default_rng(100000), hour_idx_0)
        
        for t in range(self.start_idx, self.end_idx):
            hour_index = t % 24
            lmhv = lambda_hv_data[t] * self.env.range_lmhv
            
            lp_base = lmhv * 1.5
            lp_action = lp_base * np.random.uniform(0.9, 1.1) 
            lp_base = np.clip(lp_base, self.env.lp_min, self.env.lp_max)

            di, sat_i, comp_num = self.env.User.user_response([lp_action], hour_index)
            
            total_demand = np.sum(di)
            complaint_cost = self.env.unit_costs[hour_index]*comp_num
            
            record = {
                'time_idx': t,
                'hour': hour_index,
                'lambda_hv': lmhv,
                'lambda_p': lp_action,
                'price_ratio': lp_action / (lmhv + 1e-5),
                'total_demand': total_demand,
                'complaint_cost': complaint_cost,
                'complaint_count': comp_num,
            }
            data_list.append(record)
            
        df = pd.DataFrame(data_list)
        print(f"Data collection finished. Shape: {df.shape}")
        return df

class ResponseFitter:
    """
    第二步：分布拟合
    """
    def __init__(self, dataframe):
        self.df = dataframe.copy() # 使用副本
        self.models = {}
        self.residuals_dist = {}
        
    def visualize_data(self):
        """可视化数据分布"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        sns.histplot(self.df['total_demand'], kde=True)
        plt.title('Total Demand Distribution')
        
        plt.subplot(1, 3, 2)
        sns.scatterplot(x='lambda_p', y='total_demand', data=self.df, hue='hour', palette='viridis')
        plt.title('Price vs Demand')
        
        plt.subplot(1, 3, 3)
        sns.histplot(self.df['complaint_count'], bins=range(0, int(self.df['complaint_count'].max())+2))
        plt.title('Complaint Count Distribution')
        
        plt.tight_layout()
        # [修改] 正确的保存路径拼接
        plt.savefig(os.path.join(SAVE_DIR, "Data_distribution.png"))
        plt.close()

    def fit_demand_model(self):
        """拟合负荷模型"""
        print("Fitting Demand Model...")
        
        self.df['prev_demand'] = self.df['total_demand'].shift(1)
        self.df.dropna(inplace=True)
        
        X = self.df[['hour', 'lambda_p', 'prev_demand']]
        y = self.df['total_demand']
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X, y)
        self.models['demand_regressor'] = rf
        
        y_pred = rf.predict(X)
        residuals = y - y_pred
        
        mu, std = norm.fit(residuals)
        self.residuals_dist['demand_std'] = std
        print(f"Demand Model fitted. Residual STD: {std:.4f}")
        
        # 可视化残差
        plt.figure()
        sns.histplot(residuals, kde=True)
        plt.title("Demand Residuals Distribution")
        # [修改] 正确路径
        plt.savefig(os.path.join(SAVE_DIR, "Demand_Residuals_Distribution.png"))
        plt.close()
        
        # 可视化散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(y, y_pred, alpha=0.5)
        plt.title("Demand: Actual vs Predicted")
        # [修改] show 不接受保存参数，改为 savefig
        plt.savefig(os.path.join(SAVE_DIR, "Demand_Actual_vs_Predicted.png"))
        plt.close()
        
    def fit_complaint_cost_model(self):
        """拟合投诉成本模型 (回归)"""
        print("Fitting Complaint Cost Model...")
        
        # 确保 prev_demand 存在 (如果先调了 fit_demand_model 其实已经有了，但为了安全再做一次)
        if 'prev_demand' not in self.df.columns:
            self.df['prev_demand'] = self.df['total_demand'].shift(1)
            self.df.dropna(inplace=True)
        
        # [关键修复] 特征必须与 fit_demand_model 一致，或者明确定义
        # 这里我们只用基础特征，不再用 lag_1 (投诉历史)，简化模型防止死锁
        X = self.df[['hour', 'lambda_p', 'prev_demand']]
        
        y = self.df['complaint_cost']
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X, y)
        self.models['complaint_cost_regressor'] = rf
        
        y_pred = rf.predict(X)
        residuals = y - y_pred
        
        mu, std = norm.fit(residuals)
        self.residuals_dist['complaint_cost_std'] = std
        print(f"complaint_cost Model fitted. Residual STD: {std:.4f}")
        
        # 绘图
        plt.figure(figsize=(8, 6))
        plt.scatter(y, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.title("complaint_cost: Actual vs Predicted")
        plt.savefig(os.path.join(SAVE_DIR, "complaint_cost_Actual_vs_Predicted.png"))
        plt.close()
        
        
    def save_models(self, name='user_surrogate_model.pkl'):
        save_dict = {
            'models': self.models,
            'residuals': self.residuals_dist
        }
        # [修改] 使用 os.path.join 确保路径正确
        filename = os.path.join(SAVE_DIR, name)
        with open(filename, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Models saved to {filename}")

# --- 供外部调用的代理类 ---
class SurrogateUser:
    """
    第三步：代理模型类
    """
    def __init__(self, save_dir =SAVE_DIR, model_filename='user_surrogate_model.pkl'):
        # [修改] 默认路径加上 SAVE_DIR，否则加载会失败
        model_path = os.path.join(save_dir, model_filename)
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        self.models = data['models']
        self.residuals = data['residuals']
        self.Nd = 50 
        


    def user_response_sample(self, lambda_p, hour_index, lambda_hv):
        # 初始化状态
        if not hasattr(self, 'last_agg_load'):
            self.last_agg_load = 200.0
        
        # --- 1. 预测负荷 ---
        X_demand = pd.DataFrame([[hour_index, lambda_p, self.last_agg_load]], 
                                columns=['hour', 'lambda_p', 'prev_demand'])
        
        pred_demand_mean = self.models['demand_regressor'].predict(X_demand)[0]
        
        # 采样噪声
        noise_demand = np.random.normal(0, self.residuals.get('demand_std', 1.0))
        final_demand = pred_demand_mean + noise_demand
        
        # --- 2. 预测投诉成本 ---
        # [修复] 移除 Two-Stage 逻辑，移除 Lag 特征，直接回归 Cost
        # 特征必须与 fit_complaint_cost_model 中的 X 完全一致
        X_cost = pd.DataFrame([[hour_index, lambda_p, self.last_agg_load]], 
                              columns=['hour', 'lambda_p', 'prev_demand'])
        
        pred_cost_mean = self.models['complaint_cost_regressor'].predict(X_cost)[0]
        
        # 采样噪声 (防止负数)
        if pred_cost_mean <= 0:
             # 即使预测为0，也给一点极小的随机性，防止完全死水，但主要靠 continuous value
            final_cost = max(0, np.random.normal(0, 0.1))
        else:
            noise_cost = np.random.normal(0, self.residuals.get('complaint_cost_std', 1.0))
            final_cost = max(0, pred_cost_mean + noise_cost)
        
        # --- 3. 更新状态 ---
        self.last_agg_load = final_demand
        # 注意：现在不需要维护 complaint_history 了，因为模型不再依赖它
        
        # 返回结果 (Complaints 返回的是 Cost)
        dummy_sat = np.zeros(getattr(self, 'Nd', 1)) 
        return final_demand, dummy_sat, final_cost
        
    
# ==========================================
# 主执行流程
# ==========================================

if __name__ == "__main__":
    # 1. 收集
    collector = UserResponseCollector()
    df_data = collector.collect_data()
    
    # 2. 拟合
    fitter = ResponseFitter(df_data)
    # fitter.visualize_data()
    fitter.fit_demand_model()
    fitter.fit_complaint_cost_model()
    fitter.save_models()
    
    # 3. 测试代理模型
    print("\nTesting Surrogate Model...")
    
    # 初始化环境
    env = MgComplaintFBEnv(is_user_dynamic=True, train_mode=True)
    env.reset()
    
    # 初始化代理 (此时会自动去 SAVE_DIR 找模型)
    surrogate = SurrogateUser()
    
    # 构造测试用例
    test_time_idx = 100 
    test_hour = test_time_idx % 24
    
    # 获取真实 lambda_hv 用于对比 (虽然模型输入现在不依赖它，但作为上下文打印出来也好)
    real_lmhv_val = lambda_hv_data[test_time_idx] * env.range_lmhv
    
    test_lp = 10.0 
    
    # A. 代理模型预测
    d_pred, _, c_pred = surrogate.user_response_sample(test_lp, test_hour, real_lmhv_val)
    
    print(f"Scenario: Time={test_time_idx}, Hour={test_hour}, Real HV Price={real_lmhv_val:.2f}, Test LP={test_lp}")
    print(f"[Surrogate] Demand={d_pred:.2f}, Complaints={c_pred}")
    
    # B. 验证逻辑
    print("--- Verifying against Data Collection Logic ---")
    
    # 手动同步环境状态
    env.User._random_init_user_model(np.random.default_rng(42), test_hour) 
    env.index_t = test_time_idx
    
    di_true, _, comp_true_num = env.User.user_response([test_lp], test_hour)
    d_true = np.sum(di_true)
    c_true = env.unit_costs[test_hour]*comp_true_num
    print(f"[True Env ] Demand={d_true:.2f}, Complaints={c_true:.2f}")
    
    err_demand = abs(d_pred - d_true)
    print(f"Demand Error: {err_demand:.2f} ({(err_demand/d_true)*100:.2f}%)")
    
    err_complaint_cost = abs(c_pred - c_true)
    if abs(c_true)<1.0:
        print("c_pred=",c_pred,"c_true",c_true)
    else:
        print(f"complaint_cost Error: {err_complaint_cost:.2f} ({(err_complaint_cost/(c_true))*100:.2f}%)")
    
    
    # [新增] 连续多步测试，验证 last_agg_load 是否更新
    print("\n--- Running Multi-step Verification ---")
    for i in range(3):
        h = (test_hour + i + 1) % 24
        d, _, c = surrogate.user_response_sample(test_lp, h, real_lmhv_val)
        print(f"Step +{i+1}: Demand={d:.2f}, Prev Demand Internal={surrogate.last_agg_load:.2f},complaint-cost:{c}")