import os

os.system('clear')
print('Importing Libraries:')

import numpy as np
import matplotlib.pyplot as plt

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
## CALCULATE MEAN                              #
################################################	
exp_val_0 = np.mean(preco)
print('Mean_NP = ', exp_val_0)

################################################
## CALCULATE VARIANCE                          #
################################################	
sigma2_0 = np.var(preco, ddof=1)
sigma_0 = np.sqrt(sigma2_0)
print('Var_NP = ', sigma2_0)
print('Sigma_NP = ', sigma_0)

################################################
## PLOT                                        #
################################################	
plt.plot(preco)
plt.show()
