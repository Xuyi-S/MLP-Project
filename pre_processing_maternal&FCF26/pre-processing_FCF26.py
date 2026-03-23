import pandas as pd
import numpy as np
# 1. 读入原始数据
df_raw = pd.read_csv("MLP-Project\\unicef_malawi.csv")

# 2. 复制一份分析用数据，避免直接改原始数据
df = df_raw.copy()

# 查看 FCF26 的原始分布
print(df['FCF26'].value_counts(dropna=False).sort_index())
# 查看 FCF26 的唯一值
print("Unique values in FCF26:")
print(sorted(df['FCF26'].dropna().unique()))
# 处理'NO RESPONSE'为NaN
df['FCF26_clean'] = df['FCF26'].replace('NO RESPONSE', np.nan)
# No depression=0, depression=1
fcf26_map={
    'NEVER': 0,
    'A FEW TIMES A YEAR': 1,
    'MONTHLY': 1,
    'WEEKLY': 1,
    'DAILY': 1
}
df['FCF26_binary'] = df['FCF26_clean'].map(fcf26_map)
# 查看处理后的分布
print(df['FCF26'].value_counts(dropna=False))
print(df['FCF26_clean'].value_counts(dropna=False))
print(df['FCF26_binary'].value_counts(dropna=False))