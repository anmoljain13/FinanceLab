import numpy as np
import matplotlib.pyplot as plt

S0 = 50
K = 50
T = 1
r = 0.08
sigma = 0.3

fact = lambda x : np.math.factorial(x)

def getUD(dt, ud_set):
	if ud_set == 1:
		u = np.exp(sigma * np.sqrt(dt))
		d = np.exp(-sigma * np.sqrt(dt))
	else:
		beta = 0.5 * (np.exp(-r*dt) + np.exp((r+sigma**2)*dt))	
		u = beta + np.sqrt(beta**2 - 1)
		d = beta - np.sqrt(beta**2 - 1)
	return (u,d)
	
def getQ(u,d,dt):
	return (np.exp(r*dt) - d)/(u - d)
	
def calcEuOptionVals(option, M, ud_set):
	val = [[0 for j in range(i+1)] for i in range(M+1)]
	dt = T/M
	u,d = getUD(dt, ud_set)
	q = getQ(u,d,dt)
	for i in range(M,-1,-1):
		for j in range(i,-1,-1):
			S = S0 * u**(j) * d**(i-j)
			if i == M:
				if option == 'C':
					val[i][j] = max(0, S-K)
				else:
					val[i][j] = max(0, K-S)
			else:
				val[i][j] = (np.exp(-r*dt) 
							* (q*val[i+1][j+1] + (1-q)*val[i+1][j]))
	for i in range(M,-1,-1):
		for j in range(i,-1,-1):
			val[i][j] = round(val[i][j], 2)
		
	return val
	
def calcAmPutVals(M, ud_set):
	val = [[0 for j in range(i+1)] for i in range(M+1)]
	dt = T/M
	u,d = getUD(dt, ud_set)
	q = getQ(u,d,dt)
	for i in range(M,-1,-1):
		for j in range(i,-1,-1):
			S = S0 * u**(j) * d**(i-j)
			val[i][j] = max(0, K-S)
			if i != M:
				val[i][j] = max(val[i][j], np.exp(-r*dt) 
							* (q*val[i+1][j+1] + (1-q)*val[i+1][j]))
	for i in range(M,-1,-1):
		for j in range(i,-1,-1):
			val[i][j] = round(val[i][j], 2)
	return val

def getStockPrices(M, ud_set):
	p = [[0 for j in range(i+1)] for i in range(M+1)]
	dt = T/M
	u,d = getUD(dt, ud_set)
	for i in range(M,-1,-1):
		for j in range(i,-1,-1):
			p[i][j] = S0 * u**(j) * d**(i-j)
	return p

def plotValues(M, vals, stockp, name):
	plt.figure(name)
	ax = plt.axes(projection ="3d")
	x,y,z=[[],[],[]]
	dt = T/M
	for i in range(M+1):
		t = dt*i
		for j in range(i+1):
			x.append(t)
			y.append(stockp[i][j])
			z.append(vals[i][j]) 
	ax.scatter3D(x,y,z, s=5)
	ax.set_xlabel('Time')
	ax.set_ylabel('Stock Price')
	ax.set_zlabel('Value')
	plt.title(name)
	plt.savefig('img/%s.png'%name)


	
def execSet(ud_set):
	print('Running for u d set %d' % ud_set)
	
	print('\nOption Prices:')
	print('| M | EU C | EU P | AM P |')
	print('='*26)
	for M in [5, 10, 20]:
		euc = calcEuOptionVals('C', M, ud_set)
		eup = calcEuOptionVals('P', M, ud_set)
		amp = calcAmPutVals(M, ud_set)
		print('|%s|%s|%s|%s|'%(str(M).center(3), str(euc[0][0]).center(6), 
				str(eup[0][0]).center(6), str(amp[0][0]).center(6)))
	
	euc20 = calcEuOptionVals('C', M, ud_set)
	eup20 = calcEuOptionVals('P', M, ud_set)
	amp20 = calcAmPutVals(M, ud_set)
				
	print('\n\nOption Values, M=20:')
	print('\nEuropean Call')
	print('|  t   | %s |'%('values'.center(118)))
	print('='*129)
	for t in [0, 0.25, 0.50, 0.75, 0.95]:
		i = int(t/0.05)
		print('|%s|%s|'%(str(t).center(6), str(euc20[i]).ljust(120)))
	
	print('\nEuropean Put')
	print('|  t   | %s |'%('values'.center(118)))
	print('='*129)
	for t in [0, 0.25, 0.50, 0.75, 0.95]:
		i = int(t/0.05)
		print('|%s|%s|'%(str(t).center(6), str(eup20[i]).ljust(120)))
	
	print('\nAmerican Put')
	print('|  t   | %s |'%('values'.center(118)))
	print('='*129)
	for t in [0, 0.25, 0.50, 0.75, 0.95]:
		i = int(t/0.05)
		print('|%s|%s|'%(str(t).center(6), str(amp20[i]).ljust(120)))

	stockp = getStockPrices(20,ud_set)
	plotValues(20,euc20,stockp,'European Call, Set %d'%ud_set)
	plotValues(20,eup20,stockp,'European Put, Set %d'%ud_set)
	plotValues(20,amp20,stockp,'American Put, Set %d'%ud_set)
	print('\n\n\n')
		

execSet(1)
execSet(2)
plt.show()
