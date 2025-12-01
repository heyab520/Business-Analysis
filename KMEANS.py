import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================
# 1. 讀取四張原始資料
# ============================
path = r"C:\Users\student\Documents\專題資料"

age = pd.read_excel(path + r"\age-2024(excel).xlsx")
edu = pd.read_excel(path + r"\education-2024(excel).xlsx")
job = pd.read_excel(path + r"\job-2024(excel).xlsx")
inc = pd.read_excel(path + r"\income-2024(excel).xlsx")

# 給每張表一個維度代號，讓族群名稱更好看
age["維度代號"] = "AGE"
edu["維度代號"] = "EDU"
job["維度代號"] = "JOB"
inc["維度代號"] = "INC"

dfs = [age, edu, job, inc]

for df in dfs:
    df["族群名稱"] = df["維度代號"] + "_" + df["維度內容"].astype(str)

# ============================
# 2. 把一張表轉成「族群特徵」的函數
# ============================

def make_segment_features(df):

    seg_base = df.groupby("族群名稱", as_index=False).agg(
        總金額=("信用卡交易金額", "sum"),
        總筆數=("信用卡交易筆數", "sum")
    )
    seg_base["客單價"] = seg_base["總金額"] / seg_base["總筆數"]

    # ---- 性別佔比 ----
    gender_sum = df.groupby(["族群名稱", "性別"], as_index=False)["信用卡交易金額"].sum()
    gender_pivot = gender_sum.pivot(index="族群名稱", columns="性別", values="信用卡交易金額").fillna(0)

    for g in ["男", "女"]:
        if g not in gender_pivot.columns:
            gender_pivot[g] = 0

    gender_pivot["總性別金額"] = gender_pivot["男"] + gender_pivot["女"]
    gender_pivot["男佔比"] = np.where(
        gender_pivot["總性別金額"] == 0, 0, gender_pivot["男"] / gender_pivot["總性別金額"]
    )
    gender_pivot["女佔比"] = np.where(
        gender_pivot["總性別金額"] == 0, 0, gender_pivot["女"] / gender_pivot["總性別金額"]
    )
    gender_pivot = gender_pivot[["男佔比", "女佔比"]]

    # ---- 產業別佔比 ----
    cat_sum = df.groupby(["族群名稱", "信用卡產業別"], as_index=False)["信用卡交易金額"].sum()
    cat_pivot = cat_sum.pivot(index="族群名稱", columns="信用卡產業別", values="信用卡交易金額").fillna(0)

    category_cols = ["食", "衣", "住", "行", "文教康樂", "百貨", "其他"]
    for c in category_cols:
        if c not in cat_pivot.columns:
            cat_pivot[c] = 0

    cat_total = cat_pivot[category_cols].sum(axis=1)
    cat_share = cat_pivot[category_cols].div(cat_total.replace(0, np.nan), axis=0).fillna(0)
    cat_share = cat_share.add_suffix("_佔比")

    seg_feat = seg_base.set_index("族群名稱")
    seg_feat = seg_feat.join(gender_pivot)
    seg_feat = seg_feat.join(cat_share)

    return seg_feat.reset_index()


# 四個維度各做一份族群特徵
age_seg = make_segment_features(age)
edu_seg = make_segment_features(edu)
job_seg = make_segment_features(job)
inc_seg = make_segment_features(inc)

# ============================
# 3. 合併成一張「族群矩陣」
# ============================

final = pd.concat([age_seg, edu_seg, job_seg, inc_seg], ignore_index=True)

# ============================
# 4.（新增）Elbow Method 判斷最佳 K
# ============================

feature_cols = [
    "客單價",
    "男佔比", "女佔比",
    "食_佔比", "衣_佔比", "住_佔比", "行_佔比",
    "文教康樂_佔比", "百貨_佔比", "其他_佔比"
]

X = final[feature_cols].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔥 手肘圖程式（你要的）
inertia_list = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    km.fit(X_scaled)
    inertia_list.append(km.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(K_range, inertia_list, 'o-', linewidth=2, markersize=8)
plt.title('Elbow Method（手肘法判斷最佳 K）', fontsize=14)
plt.xlabel('K 值', fontsize=12)
plt.ylabel('Inertia（慣性）', fontsize=12)
plt.grid(True)
plt.show()

# ============================
# 5. KMeans 分群（k = 4）
# ============================

k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
final["cluster"] = kmeans.fit_predict(X_scaled)

# ============================
# 6. PCA 視覺化
# ============================

pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

final["PCA1"] = pca_data[:, 0]
final["PCA2"] = pca_data[:, 1]

plt.figure(figsize=(8, 6))
plt.scatter(final["PCA1"], final["PCA2"], c=final["cluster"], cmap="tab10")
plt.colorbar()
plt.title("KMeans Clusters (Segment Profiles)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()

# === 查看 PCA1 / PCA2 各特徵權重 ===
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PCA1', 'PCA2'],
    index=feature_cols
)

print("\n===== PCA 各成分權重（Loadings）=====")
print(loadings)

# ============================
# 7. 匯出結果
# ============================

out_path = path + r"\cluster_result_segments.xlsx"
final.to_excel(out_path, index=False)

print("✅ 完成！已匯出：", out_path)
print("\n===== 各群平均特徵（只看數值欄位）=====")
num_cols = final.select_dtypes(include=[np.number]).columns
print(final.groupby("cluster")[num_cols].mean())









