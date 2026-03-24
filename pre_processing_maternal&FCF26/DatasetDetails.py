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
    child_background_vars
    + child_labour_vars
    + child_discipline_vars
    + child_functioning_vars
    + mother_background_vars
    + domestic_violence_vars
    + victimization_vars
    + marriage_union_vars
    + adult_functioning_vars
    + tobacco_alcohol_vars
    + life_satisfaction_vars
    + fertility_vars
    + household_characteristics_vars
    + insecticide_treated_net_vars
    + Water_sanitation_vars
    + handwashing_vars
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

# Child-level variables:
# Child background variables:
# CB3是孩子年龄，数值型变量
# CB4是孩子一生中是否曾经上过学/接受过学前教育从过去到现在，二元变量，存在NO RESPONSE的情况
# CB5A是孩子参加最高教育水平，有序变量，存在NO RESPONSE的情况，
# CB5B是孩子在最高教育水平上接受的年数，有序变量
# CB7是孩子在本学年有没有上学/接受学前教育，二元变量
# CB11是孩子是否有医疗保险，二元变量,存在NO RESPONSE的情况
# HH6，HH7是regionof residence，HH6二元变量（urban/rural），HH7分类变量（central/southern/northern）
# HL4是孩子的性别，二元变量
# ethnicity是孩子的民族，分类变量
# wscore是孩子的营养状况，数值型变量
# Child labour variables:
# CL2是孩子过去一周是否做过任何经济性劳动/有收入导向的劳动，二元变量(FALSE/TRUE)
# CL3是孩子过去一周从事任何经济性劳动/有收入导向的劳动的小时数，数值型变量
# CL12是孩子过去一周是否做过任何家务劳动，二元变量(FALSE/TRUE)
# CL13是孩子过去一周从事任何家务劳动的小时数，数值型变量
# Child discipline variables:
# FCD2A-K是孩子过去一个月是否经历过父母(A-K类型）体罚，二元变量，存在NO RESPONSE的情况
# FCD5 家长对体罚的态度（是否认为体罚是必要的），二元变量存在DK/NO RESPONSE的情况
# Child functioning variables:
# FCF26是孩子在过去两周内是否有抑郁特征，无序变量，存在NO RESPONSE的情况

#  Maternal variables
# Mother’sbackground
# 'WB4' 是母亲的年龄，数值型变量（存在较低异常值，15-17岁），后续处理时可以考虑剔除
# 'WB5' 是母亲是否接受过教育的变量，二元变量，存在NO RESPONSE的情况
# 'WB6A' 是母亲的最高教育水平，无序变量
# 'WB6B' 是母亲在最高教育水平上接受的年数，有序变量，存在DK的情况
#  WB14 是母亲的阅读识字能力，为有序变量，存在NO RESPONSE的情况
# Domestic violence variables:
# DV1A-E是母亲家庭暴力经历变量，二元变量，存在DK/NO RESPONSE的情况
# Victimization variables: VT1,VT9,VT20,VT21,VT22A,VT22B,VT22C,VT22D,VT22E,VT22F,VT22X
# Marriage/Union variables:
# MSTATUS是母亲的婚姻状况，无序变量，存在9.0异常值
# MA2是丈夫年龄，数值型变量，存在过小的年龄异常值(11-17岁)，后续处理时可以考虑剔除
# MA3是丈夫出轨情况，二元变量，存在NO RESPONSE的情况
# Adult functioning variables:
# disability是母亲是否有functional difficulty，二元变量
# AF10是母亲是否有记忆困难或注意力不集中，有序变量
# AF11是母亲是否有自我护理的困难，有序变量
# AF12是母亲是否有沟通的困难，有序变量，存在NO RESPONSE的情况
# Tobacco and alcohol use variables:
# TA1是母亲是否使用过烟草，二元变量，存在NO RESPONSE的情况
# TA14是母亲是否使用过酒精，二元变量，存在NO RESPONSE的情况
# Life satisfaction variables:
# LS1是母亲对当前生活的满意程度，有序变量，存在NO RESPONSE的情况
# LS2是母亲对当前生活的满意程度，有序变量，存在NO RESPONSE的情况
# LS3是与过去一年的现在相比，母亲认为当前生活是否更好，有序变量，存在NO RESPONSE的情况
# LS4是母亲对未来一年的态度，认为未来一年会更好，有序变量，存在NO RESPONSE的情况
# Fertility variables:
# CSURV是母亲存活子女数量，数值型变量
# CDEAD是母亲死亡子女数量，数值型变量

# Household Variables：
# Household characteristics variables:
# HC4是家庭住宅地板材料，无序分类变量(加上other有10个变量，但可以根据Questionnaire中的选项进行合并成4类)
# HC5是家庭住宅屋顶材料，无序分类变量(加上other有12个变量，但可以根据Questionnaire中的选项进行合并成5类)
# HC8是家庭住宅知否有电，二元变量(YES分出发电类型），存在NO RESPONSE的情况
# HC11是否有家庭成员拥有电脑，二元变量，存在NO RESPONSE的情况
# HC12是否有家庭成员拥有手机，二元变量，存在NO RESPONSE的情况
# HC13是否有家庭成员能够在家里上网，二元变量，存在NO RESPONSE的情况
# HC14是你或家庭成员是否拥有这栋房子，无序变量（own/rent/other），存在NO RESPONSE的情况
# HC15是否拥有能进行农业生产的土地，二元变量，存在NO RESPONSE的情况
# HC17是否有farm animals，二元变量，存在NO RESPONSE的情况
# HC19是否有家庭成员存在银行账户，二元变量，存在NO RESPONSE的情况
# Insecticide-treated net variables:
# TN1是家庭是否拥有蚊帐，二元变量，存在NO RESPONSE的情况
# Water and sanitation variables:
# WS1是家庭饮用水的主要来源，无序分类变量
# WS3是水源位置，无序分类变量
# WS4是取水时间，有序变量，存在(DK/MEMBERS DO NOT COLLECT/NUmber of minutes)的情况
# WS7是过去一个月是否存在缺水问题，二元变量，存在NO RESPONSE/DK的情况
# WS11是家庭马桶设施，无序分类变量，存在NO RESPONSE的情况
# WS14是家庭厕所位置，无序分类变量，存在NO RESPONSE的情况
# WS15是家庭厕所是否与非家庭成员共用，二元变量，存在NO RESPONSE的情况
# Handwashing variables:
# HW5 家里是否有肥皂洗手液，二元变量，存在NO RESPONSE的情况
