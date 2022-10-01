import numpy as np
import matplotlib.pyplot as plt 

S0=5
T=1
l = 5

np.random.seed(6)

def jumpInInterval(dt):
	return np.random.uniform() <= l*dt
	
def generatePaths(n,n_pts,mu,sigma, show_jumps=False):
	dt = T/n_pts
	for pi in range(n):
		path = [S0]
		S = S0
		jumps = []
		qp = 1
		for i in range(1,n_pts+1):
			t = i*dt
			dW = np.sqrt(dt)*np.random.normal()
			q = 1
			if jumpInInterval(dt):
				q = 1+abs(np.random.normal())/10
				jumps.append(t-dt)
				if show_jumps:
					plt.text(t-dt, S, str(round(q,2)))
			S = S + S*mu(t)*dt + sigma(t)*S*dW + (q-1)*S
			path.append(S)
		ts = [dt*i for i in range(n_pts+1)]
		plt.plot(ts, path, lw=0.5)
		plt.xlabel('t')
		plt.ylabel('S(t)')
		if show_jumps:
			plt.scatter(jumps, [path[int(t/dt)] for t in jumps], s=10, c='r')
			
plt.figure('Part 1')
plt.title('Part 1')
mu = lambda t:0.06
sigma = lambda t:0.3
generatePaths(10, 1000, mu, sigma)

plt.figure('Part 2')
plt.title('Part 2')
mu = lambda t:0.0325 - 0.25*t/T
sigma = lambda t:0.012 + 0.0138*t - 0.00125*t*t
generatePaths(10, 1000, mu, sigma)

plt.show()
	
