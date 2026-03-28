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
child_background_vars = ['CB3','CB4','CB5A','CB5B','CB7','CB11','HH6','HH7','HL4','ethnicity','wscore']
child_labour_vars = ['CL2','CL3','CL12','CL13']
child_discipline_vars = ['FCD2A','FCD2B','FCD2C','FCD2D','FCD2E','FCD2F','FCD2G','FCD2H','FCD2I','FCD2J','FCD2K','FCD5']
child_functioning_vars = ['FCF26']
mother_background_vars = ['WB4', 'WB5', 'WB6A', 'WB6B', 'WB14']
domestic_violence_vars = ['DV1A', 'DV1B', 'DV1C', 'DV1D', 'DV1E']
victimization_vars = ['VT1', 'VT9', 'VT20', 'VT21', 'VT22A', 'VT22B', 'VT22C', 'VT22D', 'VT22E', 'VT22F', 'VT22X']
marriage_union_vars = ['MSTATUS', 'MA2', 'MA3']
adult_functioning_vars = ['disability','AF10','AF11','AF12']
tobacco_alcohol_vars = ['TA1','TA14']
life_satisfaction_vars = ['LS1','LS2','LS3','LS4']
fertility_vars = ['CSURV','CDEAD']
household_characteristics_vars = ['HC4','HC5','HC8','HC11','HC12','HC13','HC14','HC15', 'HC17','HC19']
insecticide_treated_net_vars = ['TN1']
Water_sanitation_vars = [ 'WS1', 'WS3', 'WS4', 'WS7', 'WS11', 'WS14', 'WS15' ]
handwashing_vars = ['HW5']
selected_vars = (
    handwashing_vars)
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

# # 1) WB5 与 WB6A 缺失的交叉表
# ct_wb6a = pd.crosstab(df['WB5'], df['WB6A'].isna(), margins=True)
# print("WB5 vs WB6A missing:")
# print(ct_wb6a)
# # 2) WB5 与 WB6B 缺失的交叉表
# ct_wb6b = pd.crosstab(df['WB5'], df['WB6B'].isna(), margins=True)
# print("\nWB5 vs WB6B missing:")
# print(ct_wb6b)
## WB4存在NaN
## WB5: 是否上过学
df["WB5"] = df["WB5"].replace("NO RESPONSE", np.nan)
## WB6A:最高教育层级，处理结构性缺失
df.loc[df["WB5"] == "NO", "WB6A"] = "NoSchool"
## WB6B:最高教育年数。处理结构性缺失
df["WB6B"] = df["WB6B"].replace("DK", np.nan)
df.loc[df["WB5"] == "NO", "WB6B"] = '0'
## WB14:NO RESPONSE就是真实的NaN,因为所有人都要问这个问题
df["WB14"] = df["WB14"].replace("NO RESPONSE", np.nan)
cols_to_check = ['WB4',"WB5", "WB6A", "WB6B", "WB14"]
## DV1A-E 直接把NaN与NO RESPONCE改成NaN
dv_cols = ["DV1A", "DV1B", "DV1C", "DV1D", "DV1E"]
for col in dv_cols:
    df[col] = df[col].replace({
        "DK": np.nan,
        "NO RESPONSE": np.nan
    })
##  Victimization
vic_cols=['VT1', 'VT9', 'VT20', 'VT21', 'VT22A', 'VT22B', 'VT22C', 'VT22D', 'VT22E', 'VT22F', 'VT22X']
for col in vic_cols:
    df[col] = df[col].replace({
        "DK": np.nan,
        "NO RESPONSE": np.nan
    })
## Marriage/Union
# # 1) Marriage 与 MA2 缺失的交叉表
# ct_ma2 = pd.crosstab(df['MSTATUS'], df['MA2'].isna(), margins=True)
# print("MSTATUS vs MA2:")
# print(ct_ma2)
# # 2) WB5 与 WB6B 缺失的交叉表
# ct_ma3 = pd.crosstab(df['MSTATUS'], df['MA3'].isna(), margins=True)
# print("MSTATUS vs MA3:")
# print(ct_ma3)

df["MSTATUS"] = df["MSTATUS"].replace("9.0", 'Currently married/in union')
df['MA3']=df['MA3'].replace('NO RESPONSE',np.nan)
df.loc[df["MSTATUS"].isin(["Never married/in union", "Formerly married/in union"]), "MA3"] = "NotApplicable"
#结构性缺失
df['MA2']=df['MA2'].replace(['DK','NO RESPONSE'],np.nan)
df["MA2_status"] = "observed"
#在当前非婚姻状态下（从未接过婚/离过婚），MA2不适用
mask_structural = df["MSTATUS"].isin([
    "Never married/in union",
    "Formerly married/in union"
])
#非结构性缺失:当前婚姻状态下，本该有值但缺失
mask_genuine_missing = (
    (df["MSTATUS"] == "Currently married/in union") &
    (df["MA2"].isna())
)
df.loc[mask_structural, "MA2_status"] = "structural_missing"
df.loc[mask_genuine_missing, "MA2_status"] = "genuine_missing"
mar_var=['MSTATUS','MA2','MA2_status','MA3']

# Adultfunctioning 无需处理
af_var=['disability','AF10','AF11']
# Tobacco and alcoholuse
ta_var=['TA1','TA14']
for col in ta_var:
    df[col] = df[col].replace({
        "DK": np.nan,
        "NO RESPONSE": np.nan
    })
# Lifesatisfaction
ls_var=['LS1','LS2','LS3','LS4']
for col in ls_var:
    df[col] = df[col].replace({
        "DK": np.nan,
        "NO RESPONSE": np.nan
    })
# Fertility 无需处理
fer_var=['CSURV','CDEAD']

## Child-level variables
# Child_background
# # 1) CB4 与 CB5A 缺失的交叉表
# ct_cb5a = pd.crosstab(df['CB4'], df['CB5A'].isna(), margins=True)
# print("CB4 vs CB6A missing:")
# print(ct_cb5a)
# # 2) CB4 与 CB6B 缺失的交叉表
# ct_cb5b = pd.crosstab(df['CB4'], df['CB5B'].isna(), margins=True)
# print("\nCB4 vs CB6B missing:")
# print(ct_cb5b)
# # 3) CB4 与 CB7 缺失的交叉表
# ct_cb7 = pd.crosstab(df['CB4'], df['CB7'].isna(), margins=True)
# print("\nCB4 vs CB7 missing:")
# print(ct_cb7)
## CB4: 是否上过学
df["CB4"] = df["CB4"].replace("NO RESPONSE", np.nan)
## CB5A:最高教育层级，处理结构性缺失
df['CB5A']=df['CB5A'].replace("NO RESPONSE", np.nan)
df.loc[df["CB4"] == "NO", "CB5A"] = "NoSchool"
## CB5B:最高教育年数。处理结构性缺失
df.loc[df["CB4"] == "NO", "CB5B"] = '0'
## CB7:目前是否正在上学
df.loc[df["CB4"] == "NO", "CB7"] = 'NO'
## CB11
df["CB11"] = df["CB11"].replace("NO RESPONSE", np.nan)
age_child_var = ['CB3',"CB4", "CB5A", "CB5B", "CB7",'CB11']
# region 无需进行数据处理
clregion_var=['HH6','HH7','HL4','ethnicity','wscore']

# ## Childlabour
# # 1) CL2 与 CL3 缺失的交叉表
# ct_cl3 = pd.crosstab(df['CL2'], df['CL3'].isna(), margins=True)
# print("CL2 vs CL3 missing:")
# print(ct_cl3)
# # 2) CL12 与 CL13 缺失的交叉表
# ct_cl13 = pd.crosstab(df['CL12'], df['CL13'].isna(), margins=True)
# print("\nCL12 vs CL13 missing:")
# print(ct_cl13)
## 已知原本CL3与CL13就有00表示有劳动，但劳动少于1小时，
# 所以无法直接将CL2与CL12='FALSE'后的值设置为0
# 所以设立指示变量
df["CL3_status"] = "observed"
mask_structuralcl3 = df["CL2"] == False
mask_genuinecl3 = (df["CL2"] == True) & (df["CL3"].isna())
df.loc[mask_structuralcl3, "CL3_status"] = "structural_missing"
df.loc[mask_genuinecl3, "CL3_status"] = "genuine_missing"
df["CL13_status"] = "observed"
mask_structuralcl13 = df["CL12"] == False
mask_genuinecl13 = (df["CL12"] == True) & (df["CL13"].isna())
df.loc[mask_structuralcl13, "CL13_status"] = "structural_missing"
df.loc[mask_genuinecl13, "CL13_status"] = "genuine_missing"
cl_var=['CL2','CL3','CL12','CL13','CL3_status','CL13_status']
# Childdiscipline
cd_var=['FCD2A','FCD2B','FCD2C','FCD2D','FCD2E','FCD2F','FCD2G','FCD2H','FCD2I','FCD2J','FCD2K','FCD5']
for col in cd_var:
    df[col] = df[col].replace({
        "DK": np.nan,
        "NO RESPONSE": np.nan,
        "DK / NO OPINION":np.nan
    })
# Childfunctioning
df["FCF26"] = df["FCF26"].replace("NO RESPONSE", np.nan)
fcf26_map = {
    "NEVER": 'No',
    "A FEW TIMES A YEAR": 'YES',
    "MONTHLY": 'YES',
    "WEEKLY": 'YES',
    "DAILY": 'YES'
}
df['FCF26']=df['FCF26'].map(fcf26_map)

## Household variables
# Household characteristics(无结构缺失影响)
hc_var=['HC4','HC5','HC8','HC11','HC12','HC13','HC14','HC15', 'HC17','HC19']
for col in hc_var:
    df[col] = df[col].replace({
        "DK": np.nan,
        "NO RESPONSE": np.nan,
        "DK / NO OPINION":np.nan
    })
#  Insecticide treated nets
df['TN1']= df['TN1'].replace({
        "DK": np.nan,
        "NO RESPONSE": np.nan,
        "DK / NO OPINION":np.nan
    })
## Water and sanitation  ##如果要用最先用WS1与WS11，后面太复杂了
# # 1) WS1 与 WS3 缺失的交叉表
# ct_ws3 = pd.crosstab(df['WS1'], df['WS3'].isna(), margins=True)
# print("WS1 vs WS3:")
# print(ct_ws3)
# # 2) WS1 与 WS4 缺失的交叉表
# ct_ws4 = pd.crosstab(df['WS1'], df['WS4'].isna(), margins=True)
# print("WS1 vs WS4:")
# print(ct_ws4)
# # 3)WS11与WS14缺失的交叉表
# ct_ws14 = pd.crosstab(df['WS11'], df['WS14'].isna(), margins=True)
# print("WS11 vs WS14:")
# print(ct_ws14)
# # 4）WS11与WS15缺失的交叉表
# ct_ws15 = pd.crosstab(df['WS11'], df['WS15'].isna(), margins=True)
# print("WS11 vs WS15:")
# print(ct_ws15)
df[['WS1','WS11']]= df[['WS1','WS11']].replace({
        "DK": np.nan,
        "NO RESPONSE": np.nan,
        "DK / NO OPINION":np.nan
    })
## Handwashing
df['HW5']=df['HW5'].replace({"NO RESPONSE": np.nan})
df.to_csv("unicef_malawi_cleaned_full.csv", index=False)
# for col in ['HW5']:
#     print(f"\n{'='*50}")
#     print(f"Column: {col}")
#     print(f"dtype: {df[col].dtype}")
#     print(f"Number of unique values (including NaN): {df[col].nunique(dropna=False)}")
#     print(df[col].value_counts(dropna=False).sort_index().head(20))






