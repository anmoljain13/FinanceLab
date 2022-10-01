import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

S0=50
K=50
T=1
r=0.08
sigma=0.3

def tridiag(diag, n):
    A = np.zeros((n,n))
    for i in range(n):
        if i>0:
            A[i][i-1]=diag[0]
        A[i][i]=diag[1]
        if i<n-1:
            A[i][i+1]=diag[2]
    return A

def fillBoundaryCond(w, dt, dx, opt):
    nx, nt = w.shape
    q = 2*r/sigma**2
    for i in range(1,nt):
        if opt == 'C':
            w[0][i] = 0
            w[nx-1][i] = np.exp(0.5*(q+1)*nx*dx/2 + 0.25*(q+1)**2*i*dt)
        else:
            w[0][i] = np.exp(-0.5*(q-1)*nx*dx/2 + 0.25*(q-1)**2*i*dt)
            w[nx-1][i] = 0

def fillInitCond(w, dt, dx, opt):
    nx, nt = w.shape
    q = 2*r/sigma**2
    for i in range(0,nx):
        x = (i-(nx-1)/2)*dx
        if opt == 'C':
            w[i][0] = max(np.exp(x*(q+1)/2) - np.exp(x*(q-1)/2), 0)
        else:
            w[i][0] = max(np.exp(x*(q-1)/2) - np.exp(x*(q+1)/2), 0)
           

def Explicit(w, dt, dx):
    l = dt/dx**2
    nx, nt = w.shape
    A = tridiag([l, 1-2*l, l], nx-2)
    b = np.zeros((nx-2,))
    for i in range(1,nt):
        b[0]=w[0, i-1]
        b[nx-3]=w[nx-1, i-1]
        w[1:nx-1, i] = A@w[1:nx-1, i-1] + b
    return w

def Implicit(w, dt, dx):
    l = dt/dx**2
    nx, nt = w.shape
    A = tridiag([-l, 1+2*l, -l], nx-2)
    b = np.zeros((nx-2,))
    A_inv = np.linalg.inv(A)
    for i in range(1,nt):
        b[0]=w[0, i-1]
        b[nx-3]=w[nx-1, i-1]
        w[1:nx-1, i] = A_inv@ (w[1:nx-1, i-1] + b)
    return w

def CrankNicolson(w, dt, dx):
    l = dt/dx**2
    nx, nt = w.shape
    A = tridiag([-l/2, 1+l, -l/2], nx-2)
    B = tridiag([l/2, 1-l, l/2], nx-2)
    A_inv = np.linalg.inv(A)
    b = np.zeros((nx-2,))
    a = np.zeros((nx-2,))
    for i in range(1,nt):
        b[0]=w[0, i-1]
        b[nx-3]=w[nx-1, i-1]
        a[0]=w[0, i]
        a[nx-3]=w[nx-1, i] 
        w[1:nx-1, i] = A_inv@ ( B@w[1:nx-1, i-1] + b - a)
    return w

def getValue(w, t, dt, dx):
    nx, nt = w.shape
    x = np.log(S0/K)
    tau = sigma**2 * (T-t)/2
    q = 2*r/sigma**2
    xi = int(x/dx+0.5*(nx-1))
    ti = int(tau/dt)
    v = K * np.exp(-0.5*(q-1)*x - (0.25*(q-1)**2 + q)*tau) * w[xi, ti]
    return v

def blackScholes(x,t,opt):
    t = T-t
    dp = (np.log(x/K) + (r+0.5*sigma**2)*t) / (sigma * np.sqrt(t)) 
    dm = dp - sigma * np.sqrt(t)
    if opt == 'C':
        return norm.cdf(dp)*x - norm.cdf(dm)*K*np.exp(-r*t)
    else:
        return -norm.cdf(-dp)*x + norm.cdf(-dm)*K*np.exp(-r*t)

def printScheme(wC, wP, dt, dx, Scheme, name):
    print('\n\n%s'%name)
    wCS = np.array([row[:] for row in wC])
    wPS = np.array([row[:] for row in wP])
    Scheme(wCS,dt,dx)
    Scheme(wPS,dt,dx)
    print('Call:', getValue(wCS,0,dt,dx)) 
    print('Put:', getValue(wPS,0,dt,dx)) 

t_max = sigma**2*T/2
x_max = 10
t_n = 3001
x_n = 1001
dx = 2*x_max/(x_n-1)
dt = t_max/(t_n-1)

print('\nClosed Form: ')
print('Call:',blackScholes(S0, 0, 'C'))
print('Put:',blackScholes(S0, 0, 'P'))

wC=np.zeros((x_n,t_n))
fillInitCond(wC,dt,dx,'C')
fillBoundaryCond(wC,dt,dx,'C')

wP=np.zeros((x_n,t_n))
fillInitCond(wP,dt,dx,'P')
fillBoundaryCond(wP,dt,dx,'P')

printScheme(wC, wP, dt, dx, Explicit, 'Explicit:')
printScheme(wC, wP, dt, dx, Implicit, 'Implicit:')
printScheme(wC, wP, dt, dx, CrankNicolson, 'Crank Nicolson:')
