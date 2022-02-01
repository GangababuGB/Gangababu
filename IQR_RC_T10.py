import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Top_10.csv')
df = df.drop(['SILN_PROD_DESC1'], axis=1)
# df = df.astype({'SIHD_INV_DATE': np.datetime64})
freq = df.groupby(['SILN_PROD_ID'])['SIHD_INV_DATE'].count()
f = pd.DataFrame(freq).reset_index()
f_S = f.sort_values(by='SIHD_INV_DATE', ascending=False)
f_S = f_S.head(10)
values_df = f_S['SILN_PROD_ID'].unique().tolist()
print('Products available on the current Dataset: \n', values_df)

# Filtering by product
prod_buc = int(input('Create dataset for which product options listed above: \nselect any between 0-9:'))
sel_p = [values_df[prod_buc]]
FP = df[df['SILN_PROD_ID'].isin(sel_p)]
FP = FP.reset_index(drop=True)
# FP = FP.drop(['Unnamed: 0', 'SILN_PROD_ID', 'NET'], axis=1)
FP = FP.drop(['Unnamed: 0', 'SILN_PROD_ID'], axis=1)
FP = FP.astype({'SIHD_INV_DATE': np.datetime64})
FP = FP.sort_values(by='SIHD_INV_DATE', ascending=True)
FP.set_index('SIHD_INV_DATE', inplace=True)
print('Selected Product:', sel_p[0])
print(FP.head(10))
# print(FP.shape)
print(FP.columns)
# FP.to_csv(sel_p[0]+'-FP.csv')

# Resampling
print('Resampling data for time series:\n')
resampled_df = FP.resample('D')
print('\nResampling data for time series completed....\n')

# Label = 'IQR_RC_MONTH-WISE_'
# R = input('Resample Data : \nDAY[ d ] \nWeek [ w ] \nMonthly [ m ] \nSelect Process??\n>')
# if R == 'd':
#     title_csv = '-r_df_D_IMP.csv'
#     Label = 'IQR_RC_DAY-WISE_'
# elif R == 'w':
#     title_csv = '-r_df_W_IMP.csv'
#     Label = 'IQR_RC_WEEK-WISE_'
# else:
#     title_csv = '-r_df_M_IMP.csv'
#     Label = 'IQR_RC_MONTH-WISE_'

r_df = resampled_df.sum().reset_index() # sums qty repeated date
r_df.set_index('SIHD_INV_DATE', inplace=True)
r_df = r_df.replace(0, np.NaN)
import random
r_df['QTY'] = r_df['QTY'].replace(np.NaN, 0).astype(int)
POP = r_df["QTY"]
n = 0
new_list = [el for el in POP if el != n]
print(new_list)
print(len(new_list))
new_list = list(set(new_list))
new_list = sorted(new_list)
print(new_list)
print(len(new_list))
Q1 = np.quantile(new_list, 0.15)
Q3 = np.quantile(new_list, 0.60)
print(Q1)
print(Q3)
f_list = list(filter(lambda X: Q1 <= X <= Q3, new_list))
print(f_list)
random.shuffle(f_list)
new_list = f_list
print(new_list)
print(len(new_list))
# -----------------------------------------------------------
X = []
for index, row in r_df.iterrows():
    if row["QTY"] == 0:
        row["QTY"] = random.choice(new_list)
        X.append(row["QTY"])
    else:
        X.append(row["QTY"])
X = pd.DataFrame(X, index=r_df.index).astype(int)
X.columns = ['QTY2']
r_df2 = pd.concat([r_df, X], axis=1)
print(r_df2.head(20))
# -----------------------------------------------------------
title_csv = '-r_df_D_IMP.csv'
Label = 'IQR_RC_DAY-WISE_'
# r_df2.to_csv('F:/GangaBabu/03.01.2022/New_RC_IMP/Day/'+Label+sel_p[0]+title_csv)

# Resampling
print('Resampling data for time series weekly operation...\n')
title_csv = '-r_df_W_IMP.csv'
Label = 'IQR_RC_WEEK-WISE_'
r_dfw = r_df2.resample('W')
r_dfw = r_dfw.sum().reset_index()
r_dfw.set_index('SIHD_INV_DATE', inplace=True)
print(r_dfw.head(10))
print(len(r_dfw['QTY']))
# -----------------------------------------------------------
X = []
for index, row in r_dfw.iterrows():
    if row["QTY"] == 0:
        row["QTY"] = r_dfw["QTY2"]
        X.append(row["QTY"])
    else:
        X.append(row["QTY"])
print(X)
print(len(X))

X = pd.DataFrame(X, index=r_dfw.index).astype(int)
X.columns = ['QTY2']
r_dfw = pd.concat([r_dfw, X], axis=1)
print(r_dfw.head(20))
# -----------------------------------------------------------
#
# # r_dfw.to_csv('F:/GangaBabu/03.01.2022/New_RC_IMP/Week/'+Label+sel_p[0]+title_csv)
# print(r_dfw.head(10))
# print('Resampling data for time series Monthly operation...\n')
# title_csv2 = '-r_df_M_IMP.csv'
# Label2 = 'IQR_RC_MONTH-WISE_'
# r_dfm = r_df2.resample('M')
# r_dfm = r_dfm.sum().reset_index()
# r_dfm.set_index('SIHD_INV_DATE', inplace=True)
# print(r_dfm.head(10))
# # r_dfm.to_csv('F:/GangaBabu/03.01.2022/New_RC_IMP/Month/'+Label2+sel_p[0]+title_csv2)
# print('\nResampling data for time series operation completed....\n')

