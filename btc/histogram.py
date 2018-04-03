import os

os.system('clear')

#import time
import calendar
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as pltdates
#from scipy import stats

# Main program    
print('ETHUSD Return Histogram')

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
plt.plot(price_date, r)
plt.gcf().autofmt_xdate()
plt.show()

# Histogram
#hist, bin_edges = np.histogram(r, bins=200)

# Plot
plt.title('Histogram')
plt.hist(r, bins=75)
plt.show()

# Kernel Density Estimation
nbins = 75
r_hist, r_hist_edges = np.histogram(r, bins=nbins)
h = (r_hist_edges[-1], r_hist_edges[0])/nbins
x_kde = np.linspace(r_hist_edges[0], r_hist_edges[-1], nbins)

#for x in x_kde:
    