import numpy as np
from scipy.stats import norm
from scipy import optimize
import matplotlib.pyplot as plt

S0 = 50
K =50
T = 1
r = 0.08
sigma =0.3

def d(x,t=T,k=K):
    dp = (np.log(x/k) + (r+0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    dm = dp - sigma*np.sqrt(t)
    return (dp, dm)

def N(x):
    return norm.cdf(x)

def BSp(x,t,k):
    dp, dm = d(x,t,k)
    return -x*N(-dp) + k*np.exp(-r*t)*N(-dm)

def Interp(x,t):
    if not t:
        return max(K-x, 0)
    gamma = sigma**t / (1.04083*sigma**2*t + 0.009)
    Sf = K * (2*r/(sigma**2 + 2*r))**gamma
    beta = np.log(x/Sf)/np.log(K/Sf)
    alpha = (r*t/(3.9649*r*t + 0.032325)) ** beta
    return alpha * BSp(x,t,K*np.exp(r*t)) + (1-alpha) * BSp(x,t,K)    

def Quad(x,t):
    if not t:
        return max(K-x, 0)
    q = 2*r/sigma**2
    H = 1 - np.exp(-r*t)
    l = -0.5 * ((q-1) + np.sqrt((q-1)**2 + 4*q/H))
    f = lambda s : s*N(d(s)[0]) * (1-1/l) + K*np.exp(-r*t)*(1-N(d(s)[1])) - K
    Sf = optimize.newton(f,100)
    V = K - x
    if x>Sf:
        V = BSp(x,t,K) - Sf * N(d(Sf)[0]) * (x/Sf)**l / l
    return V 

print('\n\nPrice by Interpolation:', Interp(S0,T)) 
print('Price by Quadratic:', Quad(S0,T)) 

def plot(alg, name):
    plt.figure(name)
    ax = plt.axes(projection ='3d')
    S = np.linspace(2,100,25)
    T = np.linspace(0,1,25)
    S, T = np.meshgrid(S,T)
    Z = np.zeros(S.shape)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            Z[i][j] = alg(S[i][j],T[i][j])
    plt.title(name)
    ax.plot_surface(S,T,Z)
    
plot(Interp, 'Approximation by Interpolation')   
plot(Quad, 'Quadratic Approximation')   

plt.show()