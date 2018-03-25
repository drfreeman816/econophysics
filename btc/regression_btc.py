import os

os.system('clear')
print('Importing Libraries:')

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print('Simulating...')
print('')

################################################
## LE ARQUIVO                                  #
################################################
# 0:Mes, 1:Dia, 2:Ano, 3:Abertura, 4:Maxima, 
# 5:Minima, 6:Fechamento, 7:Volume, 8:MarketCap
fpt = open('btc.dat','r')
preco = np.array([])

for line in fpt:
	row = line.split()

	fech = float(row[6])
	preco = np.append(preco, fech)

preco = np.flipud(preco)		# Reverte a ordem
	
################################################
## LINEAR REGRESSION                           #
################################################	
p_range = np.array([350, 1300])
p_range = p_range - 1

print('Calculating linear regression using points ', p_range[0], ' through ', p_range[1])

x_arr = np.array(range(p_range[0], p_range[1]))
y_arr = preco[p_range[0]:p_range[1]]

a, b, r_val, p_val, std_err = stats.linregress(x_arr, y_arr)

if b < 0:
    print('y(x) = ', a, 'x', ' - ', np.abs(b))
else:
    print('y(x) = ', a, 'x', ' + ', b)

y_arr_fit = a*x_arr + b

################################################
## PLOT                                        #
################################################	
plt.plot(preco)
plt.plot(x_arr, y_arr_fit)
plt.show()
