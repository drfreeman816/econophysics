import os

os.system('clear')

import time
import calendar
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Bootstrapping
def bootstrap(arr, size, count):
    avg = np.array([])
    for i in xrange(count):
        sample = np.random.choice(arr, size=size, replace=0)
        np.append(avg, np.average(sample))
    return np.average(avg)

# Main program    
print('ETHUSD')

data_file = open('eth.dat', 'r')

price_date = []
open_price = np.array([])
high_price = np.array([])
low_price = np.array([])
close_price = np.array([])
volume = np.array([])
m_cap = np.array([])

for line in data_file:
    if not line.startswith('#'):
        row = line.split()
        # Date parsing
        month = list(calendar.month_abbr).index(row[0])
        price_date.insert(0, datetime.date(int(row[2]), month, int(row[1])))
        # Data parsing
        open_price = np.insert(open_price, 0, float(row[3]), axis=0)
        high_price = np.insert(high_price, 0, float(row[4]), axis=0)
        low_price = np.insert(low_price, 0, float(row[5]), axis=0)
        close_price = np.insert(close_price, 0, float(row[6]), axis=0)
        volume = np.insert(volume, 0, float(row[7]), axis=0)
        m_cap = np.insert(m_cap, 0, float(row[8]), axis=0)

# Dataset size
data_size = close_price.size
print ('Dataset size = ', data_size)

# Timespans
short_term = 30 # days
med_term = 120 # days
long_term = data_size # days

# Plot
plt.plot(close_price[data_size-365:data_size])
#plt.show()