# 电力与光伏数据获取完全指南 (US & EU)

*最后更新时间：2025-12-26*
*适用场景：电力市场分析、光伏仿真、强化学习 (DRL) 模型训练*

---

## 1. 快速索引：去哪里找数据？

### 🇺🇸 美国 (USA)
| 你的需求 | 推荐数据源 | 网址/操作 | 备注 |
| :--- | :--- | :--- | :--- |
| **实际电网发电量**<br>(Actual Generation) | **EIA Grid Monitor** | [eia.gov/electricity/gridmonitor](https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/US48/US48)<br>👉 点击右上角 `Download` > `Generation` | 最权威历史数据。含光伏/风电每小时实际出力 (MWh)。 |
| **PJM 市场数据**<br>(LMP, Load) | **Data Miner 2** | [dataminer2.pjm.com](https://dataminer2.pjm.com/) | 包含节点电价、负荷、容量市场数据。 |
| **光伏仿真数据**<br>(Simulation) | **NREL NSRDB** | [nsrdb.nrel.gov](https://nsrdb.nrel.gov/) | **首选**。精度 4km，物理模型，适合精确科研。需写代码下载。 |
| **光伏简易数据**<br>(不用写代码) | **Global Solar Atlas** | [globalsolaratlas.info](https://globalsolaratlas.info/) | 世界银行旗下。地图上点击即可下载 Excel，适合快速评估。 |
| **训练用数据集**<br>(Ready-to-use) | **Kaggle** | 搜 `US Solar Hourly` | 别人清洗好的 CSV，适合跑通代码。 |

### 🇪🇺 欧洲 (Europe)
| 你的需求 | 推荐数据源 | 网址/操作 | 备注 |
| :--- | :--- | :--- | :--- |
| **科研清洗数据**<br>(最推荐) | **OPSD**<br>(Open Power System Data) | [open-power-system-data.org](https://data.open-power-system-data.org/) | 德国科研机构维护。下载 `Time Series` 包，极其干净。 |
| **官方原始数据** | **ENTSO-E** | [transparency.entsoe.eu](https://transparency.entsoe.eu/) | 欧洲官方源头。界面较复杂，数据最全。 |
| **可视化图表** | **Energy-Charts** | [energy-charts.info](https://www.energy-charts.info/) | 德国 Fraunhofer 维护，图表精美，可下载。 |

---

## 2. 核心知识点与避坑

### 2.1 术语对照表 (US vs EU)
做仿真时不要混淆这两个市场的概念：

* **实时价格 (Real-Time Price)**:
    * 🇺🇸 **美国 (PJM)**: 叫 `Real-Time LMP`。有明确的实时市场交易。
    * 🇪🇺 **欧洲**: 没有统一的“实时市场”。对应的概念叫 **`Imbalance Price` (不平衡价格)**，波动极大。
* **常用基准价**:
    * 欧美通用：`Day-Ahead Price` (日前价格) 或 `Spot Price` (现货价格)。

### 2.2 光伏仿真工具对比
* **Renewables.ninja**: 界面友好，适合做欧洲或全球宏观分析。但在美国的精度（~50km）不如 NREL。
* **NREL NSRDB**: 美国本土精度最高（4km）。**做美国具体的工程项目或精细论文，必须用这个。**

### 2.3 NREL 数据下载参数选择
在下载气象数据用于计算光伏出力时，**必须勾选**：
* ✅ `GHI` (总辐射)
* ✅ `DNI` (直射辐射)
* ✅ `DHI` (散射辐射)
* ✅ `Temperature` (气温)
* ✅ `Wind Speed` (风速)
* ❌ **不要选** `Clearsky ...` (那是假设无云的理论值，不真实)。

---

