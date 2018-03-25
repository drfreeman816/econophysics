# -*- coding: utf-8 -*-
#!/usr/bin/env python2.7

__author__ = """Prof. Carlo R. da Cunha, Ph.D. <creq@if.ufrgs.br>"""

import os

os.system('clear')
print('.-------------------------------.')
print('| Read BTCBRL Data              |#')
print('| ----------------              |#')
print('|                               |#')
print('| By.: Prof. Carlo R. da Cunha  |#')
print('|                               |#')
print('|                     Mar/2018  |#')
print('\'-------------------------------\'#')
print('  ################################')
print('')
print('Importing Libraries:')

import numpy as np
import matplotlib.pyplot as pl

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
## PLOT                                        #
################################################	
pl.plot(preco)
pl.show()
