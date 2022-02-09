import os
import pandas as pd
import re
path = input(r'path1:')
directory = os.listdir(path)
df = pd.DataFrame(directory, columns=['source'])
f_name = df.source.str.split('.').str[0]
f_type = df.source.str.split('.').str[1]
df2 = pd.concat([f_name, f_type], axis=1, join='inner')
df2.columns = ['Name', 'Type']

path = input(r'path2:')
directory = os.listdir(path)
df = pd.DataFrame(directory, columns=['source'])
f_name = df.source.str.split('.').str[0]
f_type = df.source.str.split('.').str[1]
df3 = pd.concat([f_name, f_type], axis=1, join='inner')
df3.columns = ['Name', 'Type']

dd = pd.merge(df2, df3, how='inner')
print(dd)
