import pandas as pd
import numpy as np
# 1. 读入原始数据
df_raw = pd.read_csv("MLP-Project\\unicef_malawi.csv")

# 2. 复制一份分析用数据，避免直接改原始数据
df = df_raw.copy()

# 3. 先看整体数据基本情况
print("Overall dataset shape:", df.shape)
print("\nFirst 10 column names:")
print(df.columns[:10].tolist())

print("\nData types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

mother_background_vars = ['WB4', 'WB5', 'WB6A', 'WB6B', 'WB14']
domestic_violence_vars = ['DV1A', 'DV1B', 'DV1C', 'DV1D', 'DV1E']
victimization_vars = ['VT1', 'VT9', 'VT20', 'VT21', 'VT22A', 'VT22B', 'VT22C', 'VT22D', 'VT22E', 'VT22F', 'VT22X']
marriage_union_vars = ['MSTATUS', 'MA2', 'MA3']

selected_vars = (
    mother_background_vars
    + domestic_violence_vars
    + victimization_vars
    + marriage_union_vars
)

# 只保留你负责的变量
df_maternal = df[selected_vars].copy()

print("Maternal subset shape:", df_maternal.shape)
print("\nSelected columns:")
print(df_maternal.columns.tolist())

print("\nData types of selected variables:")
print(df_maternal.dtypes)

print("\nFirst 5 rows of selected variables:")
print(df_maternal.head())

# 每列缺失情况
missing_summary = pd.DataFrame({
    'missing_count': df_maternal.isna().sum(),
    'missing_percent': df_maternal.isna().mean() * 100,
    'dtype': df_maternal.dtypes
}).sort_values(by='missing_percent', ascending=False)
print(missing_summary)

# 每列取值有哪些
for col in df_maternal.columns:
    print(f"\n{'='*50}")
    print(f"Column: {col}")
    print(f"dtype: {df_maternal[col].dtype}")
    print(f"Number of unique values (including NaN): {df_maternal[col].nunique(dropna=False)}")
    print(df_maternal[col].value_counts(dropna=False).sort_index().head(20))

#处理数据

# Mother’sbackground
# 'WB5' 是母亲是否接受过教育的变量
# 处理'NO RESPONSE'为NaN
df['WB5_clean'] = df['WB5'].replace('NO RESPONSE', np.nan)
# 处理education变量'WB5'，No=0, Yes=1
wb5_map={
    'NO': 0,
    'YES': 1
}
df['WB5_binary'] = df['WB5_clean'].map(wb5_map)
# 查看处理后WB5的分布
print(f"\n{'='*50}")
print(df['WB5_binary'].value_counts(dropna=False))

# WB6A 是母亲的最高教育水平，为无序变量先不做处理
# WB6B 是母亲在最高教育水平上接受的年数，为有序变量，存在DK的情况
# 处理'DK'为NaN
df['WB6B_clean'] = df['WB6B'].replace('DK', np.nan)
# 将WB6B转换为数值型，非数值的部分会变成NaN
wb6b_map={
    'CLASS/YEAR/GRADE 1': 1,
    'CLASS/YEAR/GRADE 2': 2,
    'CLASS/YEAR/GRADE 3': 3,
    'CLASS/YEAR/GRADE 4': 4,
    'CLASS/YEAR/GRADE 5': 5,
    'CLASS/GRADE 6': 6,
    'CLASS/GRADE 7': 7,
    'CLASS/GRADE 8': 8
}
df['WB6B_numeric'] = df['WB6B_clean'].map(wb6b_map)
# 查看处理后WB6B的分布
print(f"\n{'='*50}")
print(df['WB6B_numeric'].value_counts(dropna=False))

# WB14 是母亲的阅读识字能力，为有序变量，但存在'NO RESPONSE'，先处理为NaN
df['WB14_clean'] = df['WB14'].replace('NO RESPONSE', np.nan)
# WB14存在取值'NO SENTENCE IN REQUIRED LANGUAGE / BRAILLE'特殊项，暂不处理

# Domestic violence
# 5个暴力经历变量DV1A-DV1E，存在'NO RESPONSE'与'DK'，先处理为NaN
dv_vars = ['DV1A', 'DV1B', 'DV1C', 'DV1D', 'DV1E']
for var in dv_vars:
    df[f'{var}_clean'] = df[var].replace(['NO RESPONSE', 'DK'], np.nan)
# DV1A-DV1E的取值都是YES/NO，处理为二元变量，No=0, Yes=1
dv_map={'NO': 0,
    'YES': 1}
for var in dv_vars:
    df[f'{var}_binary'] = df[f'{var}_clean'].map(dv_map)
    print(f"\n{'='*50}")
    print(f"Distribution of {var}_binary:")
    print(df[f'{var}_binary'].value_counts(dropna=False))

# Victimization
# VT1, VT9, VT20, VT21, VT22A-F, VT22X 代表不同类型的受害经历，存在'NO RESPONSE'与'DK'，先处理为NaN
# 先处理 VT1, VT9，VT22A-F, VT22X
vt_vars1 = ['VT1', 'VT9', 'VT22A', 'VT22B', 'VT22C', 'VT22D', 'VT22E', 'VT22F', 'VT22X']
for var in vt_vars1:
    df[f'{var}_clean'] = df[var].replace(['NO RESPONSE', 'DK'], np.nan)
# VT1, VT9, VT22A-F, VT22X的取值都是YES/NO，处理为二元变量，No=0, Yes=1
vt_map={'NO': 0,'YES': 1}
for var in vt_vars1:
    df[f'{var}_binary'] = df[f'{var}_clean'].map(vt_map)
    print(f"\n{'='*50}")
    print(f"Distribution of {var}_binary:")
    print(df[f'{var}_binary'].value_counts(dropna=False))
 # VT20, VT21的取值是有序变量，但存在特殊值为NEVER WALK ALONE AFTER DARK
 #也存在NO RESPONSE与DK
vt_vars2 = ['VT20', 'VT21']
for var in vt_vars2:
    df[f'{var}_clean'] = df[var].replace(['NO RESPONSE', 'DK'], np.nan)
    print(f"\n{'='*50}")
    print(f"Distribution of {var}_clean:")
    print(df[f'{var}_clean'].value_counts(dropna=False))
# 其他有序变量暂不处理，后续再进行特征工程

# Marriage/Union
# MSTATUS是婚姻状况，存在'9.0'，先处理为NaN   
df['MSTATUS_clean'] = df['MSTATUS'].replace('9.0', np.nan)
print(f"\n{'='*50}")
print(df['MSTATUS_clean'].value_counts(dropna=False))
# MSTATUS的取值是无序变量，暂不处理
# MA2是丈夫年龄 应当转换为数值型
df['MA2_clean'] = pd.to_numeric(df['MA2'], errors='coerce')
MA2_check = df.loc[df['MA2_clean'] < 18, ['MSTATUS', 'MA2_clean']].value_counts(dropna=False)
print(f"\n{'='*50}")
print(MA2_check)
# 这些低年龄并不是因为婚姻状态(MSTATUS)而导致的录入错误，要么是极端异常的个例，要不就是可疑值
# 先不处理，后续再进行处理，（方式为<18的MA2_clean值改为NaN）
# MA3是丈夫出轨经历，存在'NO RESPONSE'，先处理为NaN
df['MA3_clean'] = df['MA3'].replace('NO RESPONSE', np.nan)
# MA3的取值是YES/NO，处理为二元变量，No=0, Yes=1
ma3_map={'NO': 0,'YES': 1}
df['MA3_binary'] = df['MA3_clean'].map(ma3_map)
print(f"\n{'='*50}")
print(df['MA3_binary'].value_counts(dropna=False))
