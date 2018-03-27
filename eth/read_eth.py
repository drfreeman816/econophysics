import os

os.system('clear')

import time
import calendar
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as pltdates
from scipy import stats

# Bootstrapping
def bootstrap_avg(arr, size, count):
    avg = np.array([])
    for i in range(count):
        sample = np.random.choice(arr, size=size, replace=0)
        avg = np.append(avg, np.mean(sample))
    return np.mean(avg)

def bootstrap_var(arr, size, count):
    avg = np.array([])
    for i in range(count):
        sample = np.random.choice(arr, size=size, replace=0)
        avg = np.append(avg, np.average(sample))
    boot_avg = np.average(avg)
    boot_var = 0
    for i in range(count):
        boot_var += np.power(boot_avg-avg[i], 2)
    return boot_var/(count-1)

# Jackknife
def jackknife_avg(arr):
    size = np.size(arr)
    avg = np.array([])
    for i in range(size):
        sample = np.delete(arr, i)
        avg = np.append(avg, np.average(sample))
    return np.average(avg)

def jackknife_var(arr):
    size = np.size(arr)
    avg = np.array([])
    for i in range(size):
        sample = np.delete(arr, i)
        avg = np.append(avg, np.average(sample))
    jack_avg = np.average(avg)
    jack_var = 0
    for i in range(size):
        jack_var += np.power(jack_avg-avg[i], 2)
    return (size - 1)*jack_var/size

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
        price_date.insert(0, dt.date(int(row[2]), month, int(row[1])))
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

# Short-term analysis

print('Short-term analysis:')
print(short_term, ' days')

# Select data
x = price_date[-short_term:]
y = close_price[-short_term:]

# Usual statistics
print('Average value = ', np.average(y))
print('Variance = ', np.var(y, ddof=1))

# Bootstrapping analysis
size = int(0.7*short_term)
count = 10000
print('Bootstrapping ({1} samples of size {0}):'.format(size, count))
print('\t Average value = ', bootstrap_avg(y, size, count))
print('\t Variance = ', bootstrap_var(y, size, count))

# Jackknife analysis
print('Jackknife:')
print('\t Average value = ', jackknife_avg(y))
print('\t Variance = ', jackknife_var(y))

# Regression
print('Linear regression:')
x_days = np.array([(day - x[0]).days for day in x])
print(x_days)
a, b, r_value, p_value, std_err = stats.linregress(x_days, y)
if b < 0:
    print('y(x) = ', a, 'x', ' - ', np.abs(b))
else:
    print('y(x) = ', a, 'x', ' + ', b)
print('with r-value = {0}, p-value = {1} and std-err = {2}'.format(r_value, p_value, std_err))
y_fit = a*x_days + b

# Plot
plt.title('Short-term analysis')
plt.gca().xaxis.set_major_formatter(pltdates.DateFormatter('%d/%m/%Y'))
#plt.gca().xaxis.set_major_locator(pltdates.DayLocator())
plt.plot(x, y, 'r*')
plt.plot(x, y_fit)
plt.gcf().autofmt_xdate()
plt.show()

# Plot
plt.plot(close_price[data_size-365:data_size])
#plt.show()