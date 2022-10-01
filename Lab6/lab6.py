import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt 

S0 = 100
T = 1
Rmx = 5
nR = 501
nt = 101
dR = Rmx/(nR-1)
dt = T/(nt-1)

def CNIM(r, sig, plot=False):
    k = 0.5*dt/dR
    c = lambda x: 0.5 * sig**2 * x**2
    d = lambda x: 1 - r*x
    R = np.linspace(0, Rmx, nR)
    G = -c(R)/(2*dR**2) - d(R)/(4*dR)
    D = c(R)/(2*dR**2) + d(R)/(4*dR)
    K = c(R)/(dR**2) + 1/dt
    E = -c(R)/(dR**2) + 1/dt
    J = -c(R)/(2*dR**2) + d(R)/(4*dR)
    F = c(R)/(2*dR**2) - d(R)/(4*dR)

    A = diags([F[2:-1],E[1:-1],D[1:-2]], [-1,0,1]).toarray()
    B = diags([J[2:-1],K[1:-1],G[1:-2]], [-1,0,1]).toarray()
    b = np.zeros((nR-2,))
    H = np.zeros((nR, nt))
    H[:,nt-1] = np.maximum(1-R/T, 0)
    for i in range(nt-2,-1,-1):
        H[0,i] = (1-3*k)*H[0,i+1] + 4*k*H[1,i+1] - k*H[2,i+1]
        b[0] = F[1]*H[0,i+1] - J[1]*H[0,i]
        b[nR-3] = D[nR-2]*H[nR-1,i+1] - G[nR-2]*H[nR-1,i]
        H[1:-1,i] =  np.linalg.solve(B, A@H[1:-1,i+1] + b)

    if plot:
        X,Y = np.meshgrid(np.linspace(0,T,nt),R)
        plt.figure('H vs t and R')
        ax = plt.axes(projection ='3d')
        ax.set_title('H vs t and R (r=%.2f, sigma=%.2f)'%(r,sig))
        ax.set_xlabel('t')
        ax.set_ylabel('R')
        ax.set_zlabel('H')
        ax.plot_surface(X,Y,H)

    return S0 * H[0,0]

rs = [0.05,0.09,0.15]
sigs = [0.1,0.2,0.3]

print('\n|  r↓ sigma→ ',end='')
for sig in sigs:
    print('|%s'%str(sig).center(6), end='')
print('|')
print('='*35)
for r in rs:
    print('|%s'%str(r).center(12), end='')
    for sig in sigs:
        print('|%s'%str(round(CNIM(r,sig),2)).center(6), end='')
    print('|')

CNIM(0.05,0.3,True)
plt.show()