import os

os.system('clear')

#import time
import calendar
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as pltdates
from sklearn.neighbors import KernelDensity
from scipy import stats

# Main program    
print('BTCUSD Return Histogram')

# Read data
data_file = open('btc.dat', 'r')

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

# Define return
r = np.log(close_price) - np.log(np.roll(close_price,1))

r = np.delete(r, 0)
price_date = np.delete(price_date, 0)

# Plot
plt.title('ln(Return)')
plt.gca().xaxis.set_major_formatter(pltdates.DateFormatter('%d/%m/%Y'))
#plt.gca().xaxis.set_major_locator(pltdates.DayLocator())
plt.plot(price_date, r, linewidth=0.5)
plt.gcf().autofmt_xdate()
plt.show()

# Histogram
N = 75
plt.hist(r, bins=N, normed=True)

# Kernel Density Estimation
x_plot = np.linspace(np.min(r), np.max(r), 1000)
bw = (x_plot[-1] - x_plot[0])/N
kde = KernelDensity(kernel='epanechnikov', bandwidth=0.01).fit(r[:,np.newaxis])
plt.title('KDE Epanechnikov')
plt.xlabel('ln(return)')
plt.ylabel('P(return)')
Plog = kde.score_samples(x_plot[:,np.newaxis])
plt.plot(x_plot, np.exp(Plog), 'r', linewidth=1.0)
plt.show()

# Pareto dist analysis
plt.title('Pareto distribution analysis')
plt.xlabel('ln(return)')
plt.ylabel('ln(P(return))')
plt.plot(x_plot, Plog, linewidth=1.0)

x_fit = np.linspace(0.006, 0.08, 1000)

y_fit = kde.score_samples(x_fit[:,np.newaxis])

a, b, r_val, p_val, std_err = stats.linregress(x_fit, y_fit)

if b < 0:
    print('y(x) = ', a, 'x', ' - ', np.abs(b))
else:
    print('y(x) = ', a, 'x', ' + ', b)

y_fit = a*x_fit + b

alpha = -a-1
x_0 = np.exp((b-np.log(alpha))/alpha)
print('alpha = ', alpha)
print('x_0 = ', x_0)

plt.plot(x_fit, y_fit)

plt.show()

# Hill Estimator
alpha_H = (data_size-1)/(np.sum(np.log(np.divide(np.exp(r), x_0))))
print('Hill: alpha = ', alpha_H)

# Plot Pareto
plt.hist(np.exp(r), bins=100, normed=True)
x_plot = np.linspace(0.6, 1, 1000)
y_plot = alpha*np.power(x_0, alpha)/np.power(x_plot, 1-alpha)
y_plot_H = alpha*np.power(x_0, alpha_H)/np.power(x_plot, 1-alpha_H)
plt.plot(x_plot, y_plot, label='Fit alpha')
plt.plot(x_plot, y_plot_H, label='Hill estimator')
plt.legend()
plt.show()