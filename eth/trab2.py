import os

os.system('clear')

# Numerical libs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as pltdates
# Date parsing libs
import datetime as dt
import calendar
# Linear regression
from scipy import stats
# Kernel Density Estimation lib
from sklearn.neighbors import KernelDensity

print('Trab2 ETHUSD')

###################
#   Data parsing  #
###################

data_file = open('eth.dat', 'r')

# Data arrays
price_date = []
open_price = np.array([])
high_price = np.array([])
low_price = np.array([])
close_price = np.array([])
volume = np.array([])
m_cap = np.array([])

# Read data file line by line
for line in data_file:
    if not line.startswith('#'): # Skip comments
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

# Plot dataset
plt.title('Dataset')
plt.ylabel('USD', fontsize=16)
plt.gca().xaxis.set_major_formatter(pltdates.DateFormatter('%d/%m/%Y'))
#plt.gca().xaxis.set_major_locator(pltdates.DayLocator())
plt.plot(price_date, close_price, 'r', linewidth=0.5)
plt.gcf().autofmt_xdate()
plt.show()

# Define return
r = np.log(close_price) - np.log(np.roll(close_price, 1))
r = np.delete(r, [0,1])
price_date = np.delete(price_date, [0,1])

# Plot return
plt.title('ln(Return)')
plt.gca().xaxis.set_major_formatter(pltdates.DateFormatter('%d/%m/%Y'))
#plt.gca().xaxis.set_major_locator(pltdates.DayLocator())
plt.plot(price_date, r, linewidth=0.5)
plt.gcf().autofmt_xdate()
plt.show()

###############################################
#   Wiener-Khinchin Autocorrelation Analysis  #
###############################################

print('Wiener-Khinchin Autocorrelation Analysis:')

# FFT
r_fft = np.fft.fft(np.hanning(np.size(r))*r)
S_r = r_fft * r_fft.conjugate()

# Plot (Power Spectral Density)
plt.title('DSP')
plt.ylim((0, 1.1*np.max(S_r)))
plt.plot(S_r[:int(data_size/2)], 'r', linewidth=0.5)
plt.show()

# IFFT (Autocorrelation)
A = np.fft.ifft(S_r)
A = (1/np.max(A)) * A
# Plot
plt.title('Autocorrelation')
plt.ylim((0, 1.1*np.max(A)))
plt.plot(A[:int(data_size/2)], 'r', linewidth=0.5)
plt.show()

# Volatility Clusters
r_sq = np.square(r)
r_sq_fft = np.fft.fft(r_sq)
A_sq = np.fft.ifft(r_sq_fft * np.conj(r_sq_fft))
A_sq = (1/np.max(A_sq)) * A_sq
# Plot
plt.title('Autocorrelation Squared')
plt.ylim((0, 1.1*np.max(A_sq)))
plt.plot(A_sq[:int(data_size/2)], 'r', linewidth=0.5)
plt.show()

#################################
#   Kernel Density Estimation   #
#################################

# Histogram
print('Histogram:')
nbins = 50
print('number of bins = {}'.format(nbins))
plt.hist(r, bins=nbins, density=True)

# Kernel Density Estimation
print('Kernel Density Estimation (Epanechnikov)')
x_plot = np.linspace(np.min(r), np.max(r), 1000)
bw = (x_plot[-1] - x_plot[0])/nbins
print('bandwidth = {}'.format(bw))
kde = KernelDensity(kernel='epanechnikov', bandwidth=bw).fit(r[:,np.newaxis])
plt.title('KDE Epanechnikov')
plt.xlabel('ln(return)')
plt.ylabel('P(return)')
Plog = kde.score_samples(x_plot[:,np.newaxis])
plt.plot(x_plot, np.exp(Plog), 'r', linewidth=0.5)
plt.show()

###############################
#   Pareto Distribution Fit   #
###############################

print('Pareto Distribution Fit')

# Plot (log-log) of estimated return distribution
plt.title('Return KDE (log-log)')
plt.xlabel('ln(return)')
plt.ylabel('ln(P(return))')
plt.plot(x_plot, Plog, linewidth=1.0)

# Linear Fit on log-log plot

# Positive Returns
print('Positive Returns')

print('Linear Fit:')
# Select linear region
x_fit_p = np.linspace(0.004, 0.065, 1000)

y_fit_p = kde.score_samples(x_fit_p[:,np.newaxis])
a, b, r_val, p_val, std_err = stats.linregress(x_fit_p, y_fit_p)

if b < 0:
    print('y(x) = ', a, 'x', ' - ', np.abs(b))
else:
    print('y(x) = ', a, 'x', ' + ', b)

y_fit_p = a*x_fit_p + b

alpha_p = -a-1
x_0_p = np.exp((b-np.log(alpha_p))/alpha_p)
print('alpha = ', alpha_p)
print('x_0 = {0} (e={1})'.format(x_0_p, np.absolute(1-x_0_p)))

plt.plot(x_fit_p, y_fit_p, 'r', linewidth=0.5)

# Hill Estimator
print('Hill estimator:')
r_p = r[r > 0.0]
alpha_H_p = np.size(r_p)/np.sum(np.log(np.divide(np.exp(r_p), x_0_p)))
#x_0_H = np.exp((b-np.log(alpha_H_p))/alpha_H_p)
print('Hill: alpha = ', alpha_H_p)
print('Relative error = {}'.format(np.absolute(alpha_p-alpha_H_p)/alpha_H_p))
#print('Hill: x_0 = ', x_0_H)
#print('Relative error = {}'.format(np.absolute(x_0-x_0_H)/x_0_H))

# Negative Returns
print('Negative Returns')

print('Linear Fit:')
# Select linear region
x_fit_n = np.linspace(-0.07, -0.015, 1000)

y_fit_n = kde.score_samples(x_fit_n[:,np.newaxis])
a, b, r_val, p_val, std_err = stats.linregress(x_fit_n, y_fit_n)

if b < 0:
    print('y(x) = ', a, 'x', ' - ', np.abs(b))
else:
    print('y(x) = ', a, 'x', ' + ', b)

y_fit_n = a*x_fit_n + b

alpha_n = a-1
x_0_n = np.exp((b-np.log(alpha_n))/alpha_n)
print('alpha = ', alpha_n)
print('x_0 = {0} (e={1})'.format(x_0_n, np.absolute(1-x_0_n)))

plt.plot(x_fit_n, y_fit_n, 'r', linewidth=0.5)

# Hill Estimator
print('Hill estimator:')
r_n = np.absolute(r[r < 0.0])
alpha_H_n = np.size(r_n)/np.sum(np.log(np.divide(np.exp(r_n), x_0_n)))
print('Hill: alpha = ', alpha_H_n)
print('Relative error = {}'.format(np.absolute(alpha_n-alpha_H_n)/alpha_H_n))

# Plot everything
plt.show()

# Plot Pareto curves
plt.hist(np.exp(r), bins=nbins, density=True)

# Positive region
# Linear Fit Plot
x_plot = np.linspace(1, np.max(np.exp(r_p)), 1000)
y_plot = alpha_p * np.power(x_0_p, alpha_p) / np.power(x_plot, 1+alpha_p)
plt.plot(x_plot, y_plot, 'r', label='Linear Fit')
# Hill Estimator Plot
x_plot = np.linspace(1, np.max(np.exp(r_p)), 1000)
y_plot_H = alpha_H_p / np.power(x_plot, 1+alpha_H_p)
plt.plot(x_plot, y_plot_H, 'b', label='Hill Estimator')

# Negative region
# Linear Fit Plot
x_plot = np.linspace(1, np.max(np.exp(r_n)), 1000)
y_plot = alpha_n * np.power(x_0_n, alpha_n)/np.power(x_plot, 1+alpha_n)
plt.plot(2-x_plot, y_plot, 'r', label='Linear Fit')
# Hill Estimator Plot
x_plot = np.linspace(1, np.max(np.exp(r_n)), 1000)
y_plot_H = alpha_H_n / np.power(x_plot, 1+alpha_H_n)
plt.plot(2-x_plot, y_plot_H, 'b', label='Hill Estimator')

plt.legend()
plt.show()

#############################################################
#   Kernel Density Estimation (Absolute Value of Returns)   #
#############################################################

print('Analysis of the Absolute Value of Returns')

# Absolute Value of Returns
r_a = np.absolute(r)

# Histogram
print('Histogram:')
nbins = 50
print('number of bins = {}'.format(nbins))
plt.hist(r_a, bins=nbins, density=True)

# Kernel Density Estimation
print('Kernel Density Estimation (Epanechnikov)')
x_plot = np.linspace(np.min(r_a), np.max(r_a), 1000)
bw = (x_plot[-1] - x_plot[0])/nbins
print('bandwidth = {}'.format(bw))
kde = KernelDensity(kernel='epanechnikov', bandwidth=bw).fit(r_a[:,np.newaxis])
plt.title('KDE Epanechnikov (abs)')
plt.xlabel('ln(return)')
plt.ylabel('P(return)')
Plog = kde.score_samples(x_plot[:,np.newaxis])
plt.plot(x_plot, np.exp(Plog), 'r', linewidth=0.5)
plt.show()

###############################
#   Pareto Distribution Fit   #
###############################

print('Pareto Distribution Fit')

# Plot (log-log) of estimated return distribution
plt.title('Return KDE (log-log)')
plt.xlabel('ln(return)')
plt.ylabel('ln(P(return))')
plt.plot(x_plot, Plog, linewidth=1.0)

# Linear Fit on log-log plot

print('Linear Fit:')
# Select linear region
x_fit_a = np.linspace(0.009, 0.135, 1000)

y_fit_a = kde.score_samples(x_fit_a[:,np.newaxis])
a, b, r_val, p_val, std_err = stats.linregress(x_fit_a, y_fit_a)

if b < 0:
    print('y(x) = ', a, 'x', ' - ', np.abs(b))
else:
    print('y(x) = ', a, 'x', ' + ', b)

y_fit_a = a*x_fit_a + b

alpha_a = -a-1
x_0_a = np.exp((b-np.log(alpha_a))/alpha_a)
print('alpha = ', alpha_a)
print('x_0 = {0} (e={1})'.format(x_0_a, np.absolute(1-x_0_a)))

plt.plot(x_fit_a, y_fit_a, 'r', linewidth=0.5)

# Hill Estimator
print('Hill estimator:')
alpha_H_a = np.size(r_a)/np.sum(np.log(np.divide(np.exp(r_a), x_0_a)))
#x_0_H = np.exp((b-np.log(alpha_H_p))/alpha_H_p)
print('Hill: alpha = ', alpha_H_a)
print('Relative error = {}'.format(np.absolute(alpha_a-alpha_H_a)/alpha_H_a))
#print('Hill: x_0 = ', x_0_H)
#print('Relative error = {}'.format(np.absolute(x_0-x_0_H)/x_0_H))

plt.show()

# Plot Pareto curve
plt.hist(np.exp(r_a), bins=nbins, density=True)
# Linear Fit Plot
x_plot = np.linspace(x_0_a, np.max(np.exp(r_a)), 1000)
y_plot = alpha_a*np.power(x_0_a, alpha_a)/np.power(x_plot, 1+alpha_a)
plt.plot(x_plot, y_plot, 'r', label='Linear Fit')
# Hill Estimator Plot
y_plot_H = alpha_H_a/np.power(x_plot, 1+alpha_H_a)
x_plot = np.linspace(1, np.max(np.exp(r_a)), 1000)
plt.plot(x_plot, y_plot_H, 'k', label='Hill Estimator')

plt.legend()
plt.show()

###############################
#   Kolmogorov–Smirnov Test   #
###############################

print('Kolmogorov–Smirnov Test')

print('Linear Fit:')
print(stats.kstest(np.exp(r_a), 'pareto', args=(alpha_a, 0, 1)))
print('Hill Estimator:')
print(stats.kstest(np.exp(r_a), 'pareto', args=(alpha_H_a, 0, 1)))