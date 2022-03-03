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

s1_I=s2_I=s3_I=s4_I = 0
cnames = ['TS', 'xs1', 'xs2', 'xs3', 'xs4']
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
        dt = datetime.now()
        TS = dt.strftime("%m/%d/%Y %I:%M:%S %p")
        s1 = random.uniform(20.55, 22.55)
        s2 = random.uniform(60.55, 62.55)
        s3 = random.uniform(600.55, 620.55)
        s4 = random.uniform(90.55, 93.55)
        xs1, xs2, xs3, xs4 = round((s1+s1_I), 6), round((s2+s2_I), 6), round((s3+s3_I), 6), round((s4+s4_I), 6)
        time.sleep(1)
        s1_I=s2_I=s3_I=s4_I = 0
        print(TS, xs1, xs2, xs3, xs4)
        with open(csv_file,'a',newline='', encoding='utf-8') as f:
                csv_writer = csv.DictWriter(f, fieldnames=cnames)
                data = {
                    'TS':TS,
                    'xs1':xs1,
                    'xs2':xs2,
                    'xs3':xs3,
                    'xs4':xs4
                    }
                csv_writer.writerow(data)
