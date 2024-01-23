# -- coding: utf-8 --
# -- coding: utf-8 --
import pandas as pd
"""
筛选 陈旧性 和 继发性 肺结核d
"""
path = '../result_preprocess/TB_Count.xlsx'
df = pd.read_excel(path)

tb_secondary_info = df[df['结核类型'].str.contains('继发')]
tb_ancient_info = df[df['结核类型'].str.contains('陈旧')]
print(tb_secondary_info)
print(tb_ancient_info)
tb_secondary_info.to_excel('../result_preprocess/tb_secondary.xlsx')
tb_ancient_info.to_excel('../result_preprocess/tb_ancient.xlsx')
