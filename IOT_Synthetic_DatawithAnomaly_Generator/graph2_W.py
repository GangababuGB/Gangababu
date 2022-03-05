from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import pandas as pd
from datetime import datetime

dt = datetime.now()
dt = dt.strftime("%m.%d.%Y")


#plt.style.use('fivethirtyeight')
#plt.style.use('dark_background')
fig, (ax1,ax2) = plt.subplots(2,1)

def animate(i):
    RT = pd.read_csv('sensor_IOT.csv')
    RT['TS'] = pd.to_datetime(RT['TS'])
    TSF = RT['TS'].iloc[:1]
    TSL = RT['TS'].iloc[-1:]
    TT = str(pd.to_timedelta(TSL.values[0]-TSF.values[0]))
    data = pd.read_csv('sensor_IOT.csv')
    data = data.iloc[-60:]
    TS = data['TS'].str[-11:]
    xs1 = data['xs1']
    xs2 = data['xs2']
    xs3 = data['xs3']
    xs4 = data['xs4']
    xs5 = data['xs5']
    xs6 = data['xs6']
    xs7 = data['xs7']
    xs8 = data['xs8']  
    ax1.cla()
    
    ax1.plot(TS, xs1, linewidth=2, label = 'Sensor_1')
    ax1.plot(TS, xs2, linewidth=2, label = 'Sensor_2')
    ax1.plot(TS, xs3, linewidth=2, label = 'Sensor_3')
    ax1.plot(TS, xs4, linewidth=2, label = 'Sensor_4')
    plt.setp(ax1.get_xticklabels(),color = 'grey', fontsize = 0)
    plt.setp(ax1.get_yticklabels(), fontsize = 7) 
    ax1.set_title('IOT Live Synthetic Data : '+dt+' RunTime : '+TT, fontsize = 10)
    ax1.legend(loc = 'upper left', fontsize = 6)
    ax1.set_ylabel('Machine A', fontsize = 8)
    ax1.annotate('S1 Threshold', xy=(len(data), 23),fontsize = 7)
    ax1.axhline(23, linestyle= '--', color='grey', lw=1, alpha=0.7)
    ax1.annotate('S2 Threshold', xy=(len(data), 63),fontsize = 7)
    ax1.axhline(63, linestyle= '--', color='grey', lw=1, alpha=0.7)
    ax1.annotate('S3 Threshold', xy=(len(data), 621),fontsize = 7)
    ax1.axhline(621, linestyle= '--', color='grey', lw=1, alpha=0.7)
    ax1.annotate('S4 Threshold', xy=(len(data), 94),fontsize = 7)
    ax1.axhline(94, linestyle= '--', color='grey', lw=1, alpha=0.7)
    
    ax2.cla()
    
    ax2.plot(TS, xs5, linewidth=2, label = 'Sensor_1')
    ax2.plot(TS, xs6, linewidth=2, label = 'Sensor_2')
    ax2.plot(TS, xs7, linewidth=2, label = 'Sensor_3')
    ax2.plot(TS, xs8, linewidth=2, label = 'Sensor_4')
    plt.setp(ax2.get_xticklabels(),rotation=90, fontsize = 6)
    plt.setp(ax2.get_yticklabels(), fontsize = 7)
    ax2.legend(loc = 'upper left', fontsize = 6)
    ax2.set_xlabel('Time Stamp', fontsize = 8)
    ax2.set_ylabel('Machine B', fontsize = 8)
    ax2.annotate('S1 Threshold', xy=(len(data), 111),fontsize = 7)
    ax2.axhline(111, linestyle= '--', color='grey', lw=1, alpha=0.7)
    ax2.annotate('S2 Threshold', xy=(len(data), 351),fontsize = 7)
    ax2.axhline(351, linestyle= '--', color='grey', lw=1, alpha=0.7)
    ax2.annotate('S3 Threshold', xy=(len(data), 621),fontsize = 7)
    ax2.axhline(621, linestyle= '--', color='grey', lw=1, alpha=0.7)
    ax2.annotate('S4 Threshold', xy=(len(data), 94),fontsize = 7)
    ax2.axhline(94, linestyle= '--', color='grey', lw=1, alpha=0.7)
    plt.style.use('fivethirtyeight')
    
    fig.tight_layout()
    fig.show()

ani = FuncAnimation(plt.gcf(), animate, interval = 1000)

plt.tight_layout()
plt.show()
