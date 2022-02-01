import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
path = 'F:/GangaBabu/03.01.2022/New_RC_IMP'
d_path = path+'/Day'
w_path = path+'/Week'
m_path = path+'/Month'

path_list = os.listdir(path)
A_M = input('Automation [ 0 ] \nManual [ 1 ]  \nSelect Process??\n>')

if A_M == '0':
    raise RuntimeError("Warning ! Automation process yet to build '[•︵•]' ")
elif A_M == '1':
    print('Preforming Manual Process....   `\[•o•]/`')
    sel_path = input('Select Frequency ( d,w,m ):')
else:
    raise ReferenceError(" {@-@} \nProcess selection out of bound : Kindly enter '0' or '1'")

print(path_list)

for f in [int, float]:
    try:
        _ = f(sel_path)
    except:
        pass
    else:
        raise ValueError("Numbers not allowed")

if sel_path == 'd':
    option = d_path
    title_csv = '-r_df_D_IMP.csv'
    Label = 'DAY-WISE'
elif sel_path == 'w':
    option = w_path
    title_csv = '-r_df_W_IMP.csv'
    Label = 'WEEK-WISE'
else:
    option = m_path
    title_csv = '-r_df_M_IMP.csv'
    Label = 'MONTH-WISE'

csv_list = os.listdir(option)
csv_n = len(csv_list)
csv_list2 = pd.DataFrame(csv_list, columns=['File_Name'])
csv_list2.index.name = 'Product_No.'
# csv_list2 = np.array(csv_list)
# csv_list2 = csv_list2.reshape(csv_n, 1)
print(csv_list2)
print('Available files in current path:', len(csv_list))
n = len(csv_list)-1
fil_prod = int(input(' {•︵•}_? Select product between ( 0-' + str(n) +' ):'))

sel_p = csv_list[fil_prod]

df = pd.read_csv(option+'/'+sel_p)
print(df.head())

# for index, row in df.iterrows():
#     y1 = df['QTY'].shift(+1)
#     y2 = df['QTY']
#     # df['QTY_diff'] = (y2-y1)
#     # df['QTY_diff'] = df['QTY_diff'].fillna(0)
#     df['QTY_diff_%'] = round(((y2 - y1) / y1) * 100, 2)
#     df['QTY_diff_%'] = df['QTY_diff_%'].fillna(0)

for index, row in df.iterrows():
    y1 = df['QTY2'].shift(+1)
    y2 = df['QTY2']
    # df['QTY2_diff'] = (y2-y1)
    # df['QTY2_diff'] = df['QTY2_diff'].fillna(0)
    df['QTY2_diff_%'] = round(((y2 - y1) / y1) * 100, 2)
    df['QTY2_diff_%'] = df['QTY2_diff_%'].fillna(0)

print(df.head(20))
# df.to_csv('F:/GangaBabu/30.12.2021/Percentange_DS/'+sel_p.strip(title_csv)+'_'+str(Label)+'-Diff_Percentage.csv')

# ----------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(df['SIHD_INV_DATE'], df['QTY'], label='QTY_Org')
plt.plot(df['SIHD_INV_DATE'], df['QTY2'], label='QTY_Imp')
plt.plot(df['SIHD_INV_DATE'], df['QTY2_diff_%'], label='IMP_diff_%')
# plt.plot(df['SIHD_INV_DATE'], df['QTY_diff_%'], label='ORG_diff_%')
plt.title('CLIENT: KPL, FREQUENCY : ' + r'$\bf{' + str(Label) + '}$,' +
          ' PRODUCT QTY : ' + r'$\bf{' + str(sel_p.strip(title_csv) + '}$'))
plt.xticks(rotation=90, size=8)
plt.subplots_adjust(bottom=0.2, left=0.06, right=0.96, top=0.9)
plt.grid(color='grey', linestyle='dotted', linewidth=0.5)
plt.grid(True, color='grey', which='major', alpha=0.5)
plt.grid(True, color='grey', which='minor', alpha=0.5)
plt.xlabel('Date')
plt.ylabel('QTY')
plt.legend()
mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()
mng.window.state("zoomed")
plt.show()

df['SIHD_INV_DATE'] = pd.to_datetime(df['SIHD_INV_DATE'])
pv = df.set_index(['SIHD_INV_DATE'])
pv = pd.pivot_table(pv, index=pv.index.month, columns=pv.index.year, values='QTY2', aggfunc='sum')
month_names=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
pv.index=month_names
pv.plot.bar()
plt.title("Sum of Qty with year & month vise")
plt.xlabel("QTY by Month")
plt.ylabel("sum of Qty")
plt.xticks(rotation=90, size=8)
plt.subplots_adjust(bottom=0.2, left=0.06, right=0.96, top=0.9)
plt.grid(color='grey', linestyle='dotted', linewidth=0.5)
plt.grid(True, color='grey', which='major', alpha=0.5)
plt.grid(True, color='grey', which='minor', alpha=0.5)
mng = plt.get_current_fig_manager()
mng.window.state("zoomed")
plt.show()
