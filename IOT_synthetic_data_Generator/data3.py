import keyboard as k
import sys
import random
from datetime import datetime
import time
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import csv
import os

s1_L,s1_H = 20.55,22.55
s2_L,s2_H  = 60.55,62.55
s3_L,s3_H  = 600.55,620.55
s4_L,s4_H  = 90.55,93.55
s5_L,s5_H  = 110.55, 111.55
s6_L,s6_H  = 330.55, 350.55
s7_L,s7_H  = 600.55, 620.55
s8_L,s8_H  = 90.55, 93.55

s1_I=s2_I=s3_I=s4_I = 0 # Anomaly Initialise
s5_I=s6_I=s7_I=s8_I=label= 0 # Anomaly Initialise
cnames = ['TS', 'xs1', 'xs2', 'xs3', 'xs4', 'xs5', 'xs6', 'xs7', 'xs8', 'label']
dt = datetime.now()
dt = dt.strftime("%m.%d.%Y")      
#csv_file = 'sensor_IOT_'+dt+'.csv'
csv_file = 'sensor_IOT.csv'

with open(csv_file,'w',newline='', encoding='utf-8') as f:
    csv_writer = csv.DictWriter(f, fieldnames=cnames)
    csv_writer.writeheader()

while True:
        if k.is_pressed('1'):
                s1_I = random.uniform(5.55, 6.55)
        if k.is_pressed('2'):
                s2_I = random.uniform(5.55, 6.55)
        if k.is_pressed('3'):
                s3_I = random.uniform(50.55, 60.55)
        if k.is_pressed('4'):
                s4_I = random.uniform(5.55, 6.55)
        if k.is_pressed('5'):
                s5_I = random.uniform(5.55, 6.55)
        if k.is_pressed('6'):
                s6_I = random.uniform(5.55, 6.55)
        if k.is_pressed('7'):
                s7_I = random.uniform(50.55, 60.55)
        if k.is_pressed('8'):
                s8_I = random.uniform(5.55, 6.55)
        dt = datetime.now()
        TS = dt.strftime("%m/%d/%Y %I:%M:%S %p")
        s1 = random.uniform(s1_L,s1_H)
        s2 = random.uniform(s2_L,s2_H)
        s3 = random.uniform(s3_L,s3_H)
        s4 = random.uniform(s4_L,s4_H)
        s5 = random.uniform(s5_L,s5_H)
        s6 = random.uniform(s6_L,s6_H)
        s7 = random.uniform(s7_L,s7_H)
        s8 = random.uniform(s8_L,s8_H)
        xs1, xs2, xs3, xs4 = round((s1+s1_I), 6), round((s2+s2_I), 6), round((s3+s3_I), 6), round((s4+s4_I), 6)
        xs5, xs6, xs7, xs8 = round((s5+s5_I), 6), round((s6+s6_I), 6), round((s7+s7_I), 6), round((s8+s8_I), 6)
        if xs1>s1_H or xs2>s2_H or xs3>s3_H or xs4>s4_H or xs5>s5_H or xs6>s6_H or xs7>s7_H or xs8>s8_H:
                label = 1
        else:
                label = 0
        time.sleep(1)
        s1_I=s2_I=s3_I=s4_I = 0 # reset
        s5_I=s6_I=s7_I=s8_I = 0 # reset
        print(TS, xs1, xs2, xs3, xs4, xs5, xs6, xs7, xs8, label)
        with open(csv_file,'a',newline='', encoding='utf-8') as f:
                csv_writer = csv.DictWriter(f, fieldnames=cnames)
                data = {
                    'TS':TS,
                    'xs1':xs1,
                    'xs2':xs2,
                    'xs3':xs3,
                    'xs4':xs4,
                    'xs5':xs5,
                    'xs6':xs6,
                    'xs7':xs7,
                    'xs8':xs8,
                    'label':label
                    }
                csv_writer.writerow(data)
