# ===================================================
# 方案 B：預測 2025「客群 × 產業」金額
# - 使用：KMeans_yearly_fixed.xlsx（2014–2024）
# - Step1：預測各客群年度總金額
# - Step2：預測各客群 × 各產業「佔比」
# - Step3：相乘得到 2025 客群×產業金額
# - Step4：計算 2025 vs 2024 YoY
# ===================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ===== 0. 檔案路徑 =====
base_path = r"C:\Users\student\Documents\專題資料"
input_file = base_path + r"\KMeans_yearly_fixed.xlsx"
output_file = base_path + r"\prediction_2025_cluster_cat.xlsx"

# ===== 1. 讀取年度 KMeans 結果 =====
df = pd.read_excel(input_file)

# 預期欄位：年份, cluster, 信用卡產業別, 信用卡交易金額, cluster總金額, 產業佔比
df = df[["年份", "cluster", "信用卡產業別",
         "信用卡交易金額", "cluster總金額", "產業佔比"]].copy()

df = df.sort_values(["cluster", "信用卡產業別", "年份"])

# ===================================================
# Part A：模型1 – 預測「客群年度總金額」
# ===================================================

# 先整理成「每年 × 客群」一筆
cluster_year = (
    df.groupby(["年份", "cluster"], as_index=False)["cluster總金額"]
      .first()  # 每年的 cluster總金額 在那個檔案應該是重複的，抓 first 即可
)

# 建「前一年總金額」特徵
cluster_year = cluster_year.sort_values(["cluster", "年份"])
cluster_year["前一年總金額"] = (
    cluster_year
    .groupby("cluster")["cluster總金額"]
    .shift(1)
)

# 第一年的 NaN 用整體平均補
cluster_year["前一年總金額"] = cluster_year["前一年總金額"].fillna(
    cluster_year["前一年總金額"].mean()
)

# 準備訓練資料
X_total = cluster_year[["年份", "前一年總金額"]].copy()
y_total = cluster_year["cluster總金額"].copy()

mask_train_total = cluster_year["年份"].between(2014, 2023)
mask_test_total = cluster_year["年份"] == 2024

X_train_total = X_total[mask_train_total]
y_train_total = y_total[mask_train_total]

X_test_total = X_total[mask_test_total]
y_test_total = y_total[mask_test_total]

# 訓練 RandomForest 模型（預測客群總金額）
model_total = RandomForestRegressor(
    n_estimators=400,
    max_depth=12,
    random_state=42
)
model_total.fit(X_train_total, y_train_total)

# 評估 2024
pred_total_2024 = model_total.predict(X_test_total)
print("===== 模型1：預測客群年度總金額（2024 測試）=====")
print("R²      :", round(r2_score(y_test_total, pred_total_2024), 4))
print("RMSE    :", round(np.sqrt(mean_squared_error(y_test_total, pred_total_2024)), 2))
print("MAE     :", round(mean_absolute_error(y_test_total, pred_total_2024), 2))

# 準備 2025 要預測的客群清單：沿用 2024 出現過的客群
clusters_2024 = cluster_year[cluster_year["年份"] == 2024][["cluster"]].drop_duplicates()
clusters_2025 = clusters_2024.copy()
clusters_2025["年份"] = 2025

# 填入 2024 的總金額當作「前一年總金額」
prev_2024 = cluster_year[cluster_year["年份"] == 2024][["cluster", "cluster總金額"]]
prev_2024 = prev_2024.rename(columns={"cluster總金額": "前一年總金額"})

clusters_2025 = clusters_2025.merge(prev_2024, on="cluster", how="left")

# 如果某些 NaN，再用整體平均補
clusters_2025["前一年總金額"] = clusters_2025["前一年總金額"].fillna(
    cluster_year["cluster總金額"].mean()
)

X_2025_total = clusters_2025[["年份", "前一年總金額"]]
clusters_2025["預測_2025_cluster總金額"] = model_total.predict(X_2025_total)

# ===================================================
# Part B：模型2 – 預測「客群 × 產業 佔比」
# ===================================================

df_share = df.copy().sort_values(["cluster", "信用卡產業別", "年份"])

# 建「前一年佔比」特徵
df_share["前一年佔比"] = (
    df_share
    .groupby(["cluster", "信用卡產業別"])["產業佔比"]
    .shift(1)
)

# NaN 用全體平均佔比補
df_share["前一年佔比"] = df_share["前一年佔比"].fillna(
    df_share["產業佔比"].mean()
)

# one-hot：cluster & 產業
df_share_dum = pd.get_dummies(
    df_share[["年份", "cluster", "信用卡產業別", "前一年佔比", "產業佔比"]],
    columns=["cluster", "信用卡產業別"],
    prefix=["cluster", "cat"]
)

X_share = df_share_dum.drop(columns=["產業佔比"])
y_share = df_share_dum["產業佔比"]

mask_train_share = df_share["年份"].between(2014, 2023)
mask_test_share = df_share["年份"] == 2024

X_train_share = X_share[mask_train_share]
y_train_share = y_share[mask_train_share]

X_test_share = X_share[mask_test_share]
y_test_share = y_share[mask_test_share]

# 訓練 RandomForest 模型（預測佔比）
model_share = RandomForestRegressor(
    n_estimators=400,
    max_depth=14,
    random_state=42
)
model_share.fit(X_train_share, y_train_share)

# 評估 2024 佔比預測
pred_share_2024 = model_share.predict(X_test_share)
print("\n===== 模型2：預測客群×產業佔比（2024 測試）=====")
print("R²      :", round(r2_score(y_test_share, pred_share_2024), 4))
print("RMSE    :", round(np.sqrt(mean_squared_error(y_test_share, pred_share_2024)), 4))
print("MAE     :", round(mean_absolute_error(y_test_share, pred_share_2024), 4))

# 準備 2025 要預測的「客群 × 產業」清單
base_2024 = df_share[df_share["年份"] == 2024][
    ["cluster", "信用卡產業別", "產業佔比"]
].drop_duplicates()

base_2025 = base_2024.rename(columns={"產業佔比": "前一年佔比"}).copy()
base_2025["年份"] = 2025

# one-hot 對齊訓練用欄位
base_2025_dum = pd.get_dummies(
    base_2025[["年份", "cluster", "信用卡產業別", "前一年佔比"]],
    columns=["cluster", "信用卡產業別"],
    prefix=["cluster", "cat"]
)

for col in X_train_share.columns:
    if col not in base_2025_dum.columns:
        base_2025_dum[col] = 0

base_2025_dum = base_2025_dum[X_train_share.columns]

# 預測 2025 佔比
base_2025["預測_2025佔比_raw"] = model_share.predict(base_2025_dum)

# 負值當 0
base_2025["預測_2025佔比_raw"] = base_2025["預測_2025佔比_raw"].clip(lower=0)

# 以「客群」為單位重新正規化，讓各產業佔比加總 = 1
base_2025["sum_by_cluster"] = base_2025.groupby("cluster")["預測_2025佔比_raw"].transform("sum")
base_2025["預測_2025佔比"] = np.where(
    base_2025["sum_by_cluster"] == 0,
    0,
    base_2025["預測_2025佔比_raw"] / base_2025["sum_by_cluster"]
)

# ===================================================
# Part C：組合 – 得到 2025 客群 × 產業「預測金額」
# ===================================================

# 併上「客群年度總金額」預測
pred_2025 = base_2025.merge(
    clusters_2025[["cluster", "預測_2025_cluster總金額"]],
    on="cluster",
    how="left"
)

# 最終 2025 產業金額
pred_2025["預測_2025產業金額"] = (
    pred_2025["預測_2025_cluster總金額"] * pred_2025["預測_2025佔比"]
)

# ===================================================
# Part D：併上 2024 實際金額，算 YoY
# ===================================================

df_2024 = df[df["年份"] == 2024][
    ["cluster", "信用卡產業別", "信用卡交易金額"]
].rename(columns={"信用卡交易金額": "金額_2024"})

final = pred_2025.merge(
    df_2024,
    on=["cluster", "信用卡產業別"],
    how="left"
)

# YoY = (預測2025 - 實際2024) / 實際2024
final["YoY_2025_vs_2024"] = np.where(
    final["金額_2024"] == 0,
    np.nan,  # 避免除以 0
    (final["預測_2025產業金額"] - final["金額_2024"]) / final["金額_2024"]
)

# 整理欄位順序，方便你看
final_out = final[[
    "年份",                     # 全部是 2025
    "cluster",
    "信用卡產業別",
    "預測_2025_cluster總金額",
    "預測_2025佔比",
    "預測_2025產業金額",
    "金額_2024",
    "YoY_2025_vs_2024"
]].copy()

# 輸出 Excel
final_out.to_excel(output_file, index=False)
print("\n✅ 已輸出預測檔：", output_file)
