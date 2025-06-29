import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. 读取数据
data = pd.read_excel("data/class_test.xlsx")

# 2. 简化决策策略（可根据需要精细分类）
def simplify_strategy(text):
    if pd.isna(text):
        return "缺失"
    if "自己" in text or "我" in text:
        return "自我主导"
    if "ai" in text.lower() or "模型" in text or "大模型" in text:
        return "信任AI"
    if "一起" in text or "合作" in text:
        return "合作"
    return "其他"

data["简化策略"] = data["群体协商决策策略"].apply(simplify_strategy)

# 3. 编码分类变量
label_cols = ["回忆方式", "简化策略"]
for col in label_cols:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# 4. 选择聚类特征（包含小组人数、回忆方式、简化策略）
features = data[["小组人数", "回忆方式", "简化策略"]].copy()
features = features.fillna(0)  # 防止聚类报错

# 5. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 6. 聚类分析（默认设定为3类，可调整）
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

data["聚类标签"] = clusters

# 7. 聚类可视化（PCA降维）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data["聚类标签"], palette="Set2")
plt.title("聚类结果可视化（PCA降维）")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="聚类编号")
plt.tight_layout()
plt.show()

# 8. 分析各聚类平均得分
print("\n各聚类平均得分（得分率）:")
print(data.groupby("聚类标签")["得分率"].mean())

# 可选：输出到文件以便报告中引用
data.to_excel("data/聚类分析结果输出.xlsx", index=False)