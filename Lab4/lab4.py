import numpy as np
import matplotlib.pyplot as plt

S0 = 50
K = 50
r = 0.08
sigma = 0.3
T = 1

t_max = sigma**2*T/2
x_max = 10
t_n = 1001
x_n = 501
dx = 2*x_max/(x_n-1)
dt = t_max/(t_n-1)

q = 2*r/sigma**2
l = dt/dx**2
th = 0.5
er = 1e-9

def g_func(x,t):
    mx = max(np.exp(0.5*(q-1)*x) - np.exp(0.5*(q+1)*x), 0)
    return np.exp(0.25*((q-1)**2 + 4*q)*t) * mx

def SOR(v, b, g):
    vn = np.zeros(v.shape)
    vn[0]=0
    v[-1]=0
    wr = 1.01
    while True:
        for i in range(x_n-2):
            p = (b[i] + l*th * (vn[i] + v[i+2])) / (1 + 2*l*th)
            vn[i+1] = max(g[i+1], v[i+1] + wr*(p - v[i+1]))
        if np.linalg.norm(v-vn) <= er:
            break        
        v = vn
    return v

w = np.zeros((x_n, t_n))
w[:,0] = [g_func(-x_max + i*dx, 0) for i in range(x_n)]

for i in range(t_n-1):
    # print(i, w[int(0.5*(x_n-1)),i])
    g = [g_func(-x_max + j*dx, (i+1)*dt) for j in range(x_n)]
    b = np.zeros(x_n)
    for j in range(2, x_n-1):
        b[j-1] = w[j,i] + l*(1-th)*(w[j+1, i] - 2*w[j,i] + w[j-1, i])
    b[0] = w[1,i] + l*(1-th)*(w[2, i] - 2*w[1,i] + g_func(0,i*dt)) + l*th*g[0]
    b[x_n-3] = w[x_n-2, i] + l*(1-th)*(w[x_n-3, i] - 2*w[x_n-2, i] + g_func(x_max,i*dt)) + l*th*g[x_n-1]
    v = np.maximum(w[:,i], g)
    w[:, i+1] = SOR(v, b, g)

x = np.log(S0/K)
tau = sigma**2 * T/2
xi = int(x/dx+0.5*(x_n-1))
ti = int(tau/dt)

print(K * np.exp(-0.5*(q-1)*x - (0.25*(q-1)**2 + q)*tau) * w[xi, ti])