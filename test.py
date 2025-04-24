import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from scipy.stats import rankdata

# 1. 加载数据
file_path = "kuka_axis_run_info_1345949_A1.csv"  # 替换为你的数据路径
data = pd.read_csv(file_path)
data["time"] = pd.to_datetime(data["collect_time"])
data["hour"] = data["time"].dt.hour
data["dayofweek"] = data["time"].dt.dayofweek
data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)

# 2. 选择特征列（假设与健康相关的主要特征）
features = ["torque", "temperature", "current"]
X = data[features]
# 3. 数据标准化（重要：使特征具有相同的尺度）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 初始化 Isolation Forest 模型
isolation_forest = IsolationForest(
    n_estimators=100,  # 决策树的数量
    contamination=0.000001,  # 假定异常点的比例为 0.0001%
    random_state=42,
)

# 5. 训练模型并预测
# data['anomaly_score'] = isolation_forest.fit_predict(X_scaled)

# 2. 训练模型
isolation_forest.fit(X_scaled)
model_filename = "isolation_forest_model.joblib"
dump(isolation_forest, model_filename)

# load
isolation_forest = load("isolation_forest_model.joblib")

data["anomaly_score"] = isolation_forest.predict(X_scaled)

scores = isolation_forest.decision_function(X_scaled)  #
data["health_status"] = np.where(data["anomaly_score"] == -1, "faulty", "healthy")

# # 将健康分值转换为百分制
# min_score = scores.min()
# max_score = scores.max()
# health_score_percentage = (scores - min_score) / (max_score - min_score) * 100


robust_scaler = RobustScaler(with_centering=True, with_scaling=True)
robust_scaler.fit(scores.reshape(-1, 1))
scaled_scores = robust_scaler.transform(scores.reshape(-1, 1)).flatten()


data["unadjust_health_score_percentage"] = scaled_scores


def percentile_scoring(scores, base_scores):
    """
    基于从小到大排序的百分位分段评分
    """
    # 对base_scores进行升序排序
    sorted_base = np.sort(base_scores)[::-1]
    n = len(sorted_base)

    # 计算百分位点
    p99 = sorted_base[int(n * 0.9999)]  # 前99.9%分界点
    p99_99 = sorted_base[int(n * 0.99999)]  # 前99.99%分界点

    # 初始化结果数组
    result = np.zeros_like(scores)

    # 分段评分
    top1_mask = scores >= p99
    next0_99_mask = (scores < p99) & (scores >= p99_99)
    bottom0_01_mask = (scores < p99_99) & (scores >= sorted_base.min())
    below_min_mask = scores < sorted_base.min()

    # 前1%: 90-100分
    result[top1_mask] = 90 + (scores[top1_mask] - p99) / (sorted_base.max() - p99) * 10

    # 接下来的0.99%: 60-90分
    result[next0_99_mask] = 60 + (scores[next0_99_mask] - p99_99) / (p99 - p99_99) * 30

    # 剩下的0.01%: 20-50分
    result[bottom0_01_mask] = (
        20
        + (scores[bottom0_01_mask] - sorted_base.min())
        / (p99_99 - sorted_base.min())
        * 30
    )

    # 比最小值还小的: 0-10分
    result[below_min_mask] = (scores[below_min_mask] / sorted_base.min()) * 10

    return np.clip(result, 0, 100)

    # Use sigmoid function to adjust the distribution


def adjusted_sigmoid(x, center=0.15, steepness=10):
    """
    Adjusted sigmoid function, which controls the center and steepness
    center: The center point (0-1), the smaller it is, the higher the overall score
    steepness: Steepness, the larger the value, the steeper the curve
    """
    return 100 / (1 + np.exp(-steepness * (x / 100 - center)))


# 基于排名进行重映射
def percentile_remap(scores, target_median=55):
    """
    将分数重新映射，使得中位数达到目标值
    scores: 原始分数
    target_median: 期望的中位数(0-100)
    """
    ranks = rankdata(scores) - 1  # 从0开始的排名
    n = len(scores)

    # 计算目标分布
    # 此处使用指数函数创建偏向高分的分布
    target_percentiles = np.exp(np.log(target_median / 100) * (1 - ranks / n)) * 100

    return target_percentiles


# health_score_percentage = percentile_remap(health_score_percentage)


def segment_mapping(scores):
    """
    使用分段函数映射分数
    分段函数可以针对不同分数区间采用不同的转换策略
    """
    result = np.zeros_like(scores)

    # 将分数分为三个区间
    very_low_mask = scores < 1
    low_mask = (scores >= 1) & (scores < 10)
    high_low_mask = (scores >= 10) & (scores < 40)
    mid_mask = (scores >= 40) & (scores < 60)
    mid_high_mask = (scores >= 60) & (scores < 85)
    high_mask = scores >= 85

    # 低分区间: 提升到10 - 20范围
    result[very_low_mask] = 10 + (scores[very_low_mask]) * 20

    # 低分区间: 提升到60-65范围
    result[low_mask] = 60 + (scores[low_mask] / 10) * 5

    # 低分区间: 提升到65-70范围
    result[high_low_mask] = 65 + ((scores[high_low_mask] - 10) / 30) * 5

    # 中分区间: 提升到70-80范围
    result[mid_mask] = 70 + ((scores[mid_mask] - 40) / 20) * 10

    # 中分区间: 提升到80-90范围
    result[mid_high_mask] = 80 + ((scores[mid_high_mask] - 60) / 26) * 10

    # 高分区间: 提升到90-100范围
    result[high_mask] = 90 + ((scores[high_mask] - 85) / 16) * 10

    return result


# health_score_percentage = segment_mapping(health_score_percentage)
health_score_percentage = percentile_scoring(scores, scores)
data["health_score_percentage"] = health_score_percentage

# 6. 检查异常点数量和分布
anomaly_count = data["health_status"].value_counts()

# 7. 保存结果asfsad
output_file = "robot_health_model_output.csv"  # 输出文件路径
data.to_csv(output_file, index=False)

# 打印结果统计信息
# print(f"异常点数量: {anomaly_count['faulty']} / {len(data)}")
print(f"输出文件保存至: {output_file}")

# 可视化：绘制某个特征的分布及异常点
plt.figure(figsize=(10, 6))
plt.scatter(data["time"], data["health_score_percentage"], alpha=0.6)
plt.xlabel("time")
plt.ylabel("health_score_percentage")
plt.title("Torque with Health Status")
plt.show()
