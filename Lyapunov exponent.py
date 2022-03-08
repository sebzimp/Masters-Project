# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:50:07 2022

@author: sebzi
"""

from numba import njit, cfunc, jit
from NumbaLSODA import lsoda_sig, lsoda

import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

#parameters
a1 = 0
b1= 0.495
M1 = 2.05*10**10
a2 = 7.258
b2 = 0.520
M2 = 25.47*10**10
qa = 1.2
qb = 0.9
Ohm = 60
G = 4.3009*10**(-6)


#potential
def potential(x,y,z,PARAMETERS = [a1,b1,M1,a2,b2,M2,qa,Ohm,G]):
    
    
    V = -G*M1*(x**2 + y**2/qa**2 + (a1 + (z**2/qb**2 + b1**2)**(0.5)  )**2)**(-0.5) - G*M2* (x**2 + y**2/qa**2 + (a2 + (z**2/qb**2 + b2**2)**(0.5)   )**2 )**(-0.5) 
    return V

#Hamiltonian
def Hamiltonian(x,y,px,py):
  
    H = 0.5*(px**2 + py**2) + potential(x,y,0) -Ohm*(x*py - y*px)
    return H

@jit
def dp2dx2(x,y):
    
    f = G*M1*(x**2 + y**2/qa**2 + (a1 +b1 )**2)**(-1.5) - 3*G*M1*x**2*(x**2 + y**2/qa**2 + (a1 +b1 )**2)**(-2.5) + \
        G*M2*(x**2 + y**2/qa**2 + (a2 +b2 )**2)**(-1.5) - 3*G*M2*x**2*(x**2 + y**2/qa**2 + (a2 +b2 )**2)**(-2.5)
    
    return f

@jit
def dp2dxdy(x,y):
    
    f = - 3*G*M1*x*y/qa**2*(x**2 + y**2/qa**2 + (a1 +b1 )**2)**(-2.5) - 3*G*M2*x*y/qa**2*(x**2 + y**2/qa**2 + (a2 +b2 )**2)**(-2.5)
    
    return f

@jit
def dp2dy2(x,y):
    
    f =  G*M1/qa**2*(x**2 + y**2/qa**2 + (a1 +b1 )**2)**(-1.5) - 3*G*M1*y**2/qa**4*(x**2 + y**2/qa**2 + (a1 +b1 )**2)**(-2.5) + \
        G*M2/qa**2*(x**2 + y**2/qa**2 + (a2 +b2 )**2)**(-1.5) - 3*G*M2*y**2/qa**4*(x**2 + y**2/qa**2 + (a2 +b2 )**2)**(-2.5)
    
    return f

#equation of motion
@cfunc(lsoda_sig)
def Hz0(t, a , da,p): #a,t

    da[0] = a[2] + Ohm*a[1]
    
    da[1] =  a[3] - Ohm*a[0]
    
    da[2] = Ohm*a[3] - a[0]*G* ( M1* (a[0]**2 + a[1]**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (a[0]**2 + a[1]**2/ qa**2 + (a2+ b2 )**2 )**(-1.5) )
    
    da[3] =  -Ohm*a[2] - (a[1]/qa**2) *G* ( M1* (a[0]**2 + a[1]**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (a[0]**2 + a[1]**2/ qa**2 +  (a2 + b2 )**2)**(-1.5) )

    da[4] = Ohm*a[5] + a[6] #variation of x
    
    da[5] = -Ohm*a[4] + a[7] #variation of y
    
    da[6] = - dp2dx2(a[0],a[1])* a[4] - dp2dxdy(a[0],a[1])*a[5] + Ohm*a[7] #variation of px
    
    da[7] = - dp2dy2(a[0],a[1])* a[5] - dp2dxdy(a[0],a[1])*a[4] - Ohm*a[6] #variation of py 

def norm(x,y,px,py):
    norm = np.sqrt( x**2.0 + y**2.0 + px**2.0 + py**2.0 )
    return norm

#Energy
A = 43950
H0 = -4.2*A #hamiltonian


#integration time
T = 1000000 #timesteps
t = np.linspace(0.0, 1000.0, T)


delta = 10.0**(-7.0)

x0 = 0
y0 = 1
py0 = 0

delta = Ohm**2 *y0**2 - 2*(0.5*py0**2 + potential(x0,y0,0) - H0) 

px0 = -Ohm*y0 + np.sqrt(delta)

delx0 = delta*x0
dely0 = delta*y0
delpx0 = delta*px0
delpy0 = delta*py0


mCLE = []
plott= []

funcptr = Hz0.address # address to ODE function for


u0 = np.array([x0,y0,px0,py0, delx0,dely0, delpx0,delpy0]) # Initial conditions

          
usol, success = lsoda(funcptr, u0, t,rtol = 1.0e-8, atol = 1.0e-9) #solving EoM

    
for i in range(50, len(t)):
    
    mCLE.append( 1.0/(t[i]) * np.log(norm(usol[i][4],usol[i][5],usol[i][6],usol[i][7])/ norm(delx0,dely0,delpx0,delpy0)))
    
    plott.append(t[i])

                               
end = time.time()
print(end - start)

plt.plot(plott,mCLE)
plt.xscale("log")
plt.yscale("log")



