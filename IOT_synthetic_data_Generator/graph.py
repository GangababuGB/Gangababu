from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import pandas as pd

plt.style.use('fivethirtyeight')

x_vals = []
y_vals = []
#TS, xs1, xs2, xs3, xs4
index = count()
def animate(i):
    data = pd.read_csv('sensor_IOT.csv')
    data = data.iloc[-60:]
    TS = data['TS']
    xs1 = data['xs1']
    xs2 = data['xs2']
    xs3 = data['xs3']
    xs4 = data['xs4']
    
    plt.cla()
    
    plt.plot(TS, xs1, linewidth=1, label = 'Sensor_1')
    plt.plot(TS, xs2, linewidth=1, label = 'Sensor_2')
    plt.plot(TS, xs3, linewidth=1, label = 'Sensor_3')
    plt.plot(TS, xs4, linewidth=1, label = 'Sensor_4')
    plt.xticks(rotation=90, fontsize = 6)

    plt.legend(loc = 'upper left', fontsize = 6)
    plt.tight_layout()

ani = FuncAnimation(plt.gcf(), animate, interval = 1000)

plt.tight_layout()
plt.show()
