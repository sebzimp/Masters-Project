# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:24:59 2022

@author: sebzi
"""

from numba import njit, cfunc, jit, uint64

import numpy as np
import matplotlib.pyplot as plt
import time

import decimal
decimal.getcontext().prec = 100

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
#@njit(fastmath=True)
def potential(x,y,z,PARAMETERS = [a1,b1,M1,a2,b2,M2,qa,Ohm,G]):
    
    
    V = -G*M1*(x**2 + y**2/qa**2 + (a1 + (z**2/qb**2 + b1**2)**(0.5)  )**2)**(-0.5) - G*M2* (x**2 + y**2/qa**2 + (a2 + (z**2/qb**2 + b2**2)**(0.5)   )**2 )**(-0.5) 
    return V

#Hamiltonian
#@jit
def Hamiltonian(x,y,px,py):
  
    H = 0.5*(px**2 + py**2) + potential(x,y,0) -Ohm*(x*py - y*px)
    return H

#@jit
def dp2dx2(x,y):
    
    f = G*M1*(x**2 + y**2/qa**2 + (a1 +b1 )**2)**(-1.5) - 3*G*M1*x**2*(x**2 + y**2/qa**2 + (a1 +b1 )**2)**(-2.5) + \
        G*M2*(x**2 + y**2/qa**2 + (a2 +b2 )**2)**(-1.5) - 3*G*M2*x**2*(x**2 + y**2/qa**2 + (a2 +b2 )**2)**(-2.5)
    
    return f

#@jit
def dp2dxdy(x,y):
    
    f = - 3*G*M1*x*y/qa**2*(x**2 + y**2/qa**2 + (a1 +b1 )**2)**(-2.5) - 3*G*M2*x*y/qa**2*(x**2 + y**2/qa**2 + (a2 +b2 )**2)**(-2.5)
    
    return f

#@jit
def dp2dy2(x,y):
    
    f =  G*M1/qa**2*(x**2 + y**2/qa**2 + (a1 +b1 )**2)**(-1.5) - 3*G*M1*y**2/qa**4*(x**2 + y**2/qa**2 + (a1 +b1 )**2)**(-2.5) + \
        G*M2/qa**2*(x**2 + y**2/qa**2 + (a2 +b2 )**2)**(-1.5) - 3*G*M2*y**2/qa**4*(x**2 + y**2/qa**2 + (a2 +b2 )**2)**(-2.5)
    
    return f


#@jit
def Hz0(a): #a,t

    dx = a[2] + Ohm*a[1]
    
    dy =  a[3] - Ohm*a[0]
    
    dpx = Ohm*a[3] - a[0]*G* ( M1* (a[0]**2 + a[1]**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (a[0]**2 + a[1]**2/ qa**2 + (a2+ b2 )**2 )**(-1.5) )
    
    dpy =  -Ohm*a[2] - (a[1]/qa**2) *G* ( M1* (a[0]**2 + a[1]**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (a[0]**2 + a[1]**2/ qa**2 +  (a2 + b2 )**2)**(-1.5) )

    dvarx = Ohm*a[5] + a[6] #variation of x
    
    dvary = -Ohm*a[4] + a[7] #variation of y
    
    dvarpx = - dp2dx2(a[0],a[1])* a[4] - dp2dxdy(a[0],a[1])*a[5] + Ohm*a[7] #variation of px
    
    dvarpy = - dp2dy2(a[0],a[1])* a[5] - dp2dxdy(a[0],a[1])*a[4] - Ohm*a[6] #variation of py 
    
    return  np.array([dx,dy,dpx,dpy,dvarx,dvary,dvarpx,dvarpy,dvarx,dvary,dvarpx,dvarpy]) 


#@jit
def norm(a):
    norm = np.sqrt( a[0]**2.0 + a[1]**2.0 + a[2]**2.0 + a[3]**2.0 )
    return norm

#@jit
def RK6(time,h,a0):
    
    t = np.arange(0,time, h)
#    sol = [a0]
    
    SALI = []
    
    tot = int(len(t))
    
    an = a0
    for i in range(0,tot):
        

        k1 = h*Hz0(an)
        
        k2 = h*Hz0(an +k1)
        
        k3 = h*Hz0(an + (3*k1+k2)/8)
        
        k4 = h*Hz0(an + (8*k1+2*k2+8*k3)/27)
        
        k5 = h*Hz0(an+ (3*(3*21**0.5 -7)*k1 -8*(7-21**0.5)*k2 +48*(7 - 21**0.5)*k3 - 3*(21 - 21**0.5)*k4 )/392)
        
        k6 = h*Hz0(an + (-5*k1*(231+ 51*(21)**0.5) -40*k2*(7 + 21**0.5) -320*k3*21**0.5 + 3*k4*(21+121*21**0.5) \
                         + 392*k5*(6 + 21**0.5) )/1960)
        
        k7 = h*Hz0(an + (15*k1*(22+7*21**0.5) + 120*k2 +40*k3*(7*21**0.5 -5) -63*k4*(3*21**0.5-2) \
                         -14*k5*(49 + 9*21**0.5) + 70*k6*(7- 21**0.5)  )/180 )
        
        anext = an + (9*k1 + 64*k3 + 49*k5 + 49*k6 + 9*k7)/180
        
        #sol.append(anext)
        
        #normalisation attempt
        w1 = np.array([anext[4],anext[5],anext[6],anext[7] ])
    
        normal = norm([anext[4],anext[5],anext[6],anext[7]])
        

        w1hat = np.divide(w1,normal)
        
        w2 = np.array([anext[8],anext[9],anext[10],anext[11] ])
    
        normal = norm([anext[8],anext[9],anext[10],anext[11] ])
        

        w2hat = np.divide(w2,normal)
        
        sal = min(norm(w1hat+w2hat), norm(w1hat-w2hat))
        
        SALI.append(sal)

        an = [anext[0],anext[1],anext[2],anext[3], w1hat[0], w1hat[1], w1hat[2], w1hat[3], w2hat[0], w2hat[1], w2hat[2], w2hat[3]]
            
    return t, SALI        #, sol


#@njit(fastmath=True)


A = 43950
H0 = -4.2*A #hamiltonian

x0 = 0

delx0 = 0.28213825
dely0 = 0.70534562
delpx0 = 0.14106912
delpy0 = 0.63481105   

delx1 = -0.05819898
dely1 = -0.31348434
delpx1 = 0.93300721
delpy1 = 0.16684722


y0 = 1
py0 = 0

delta = Ohm**2 *y0**2 - 2*(0.5*py0**2 + potential(x0,y0,0) - H0) 
px0 = -Ohm*y0 + np.sqrt(delta)
   
b = RK6(100,0.0003, np.array([x0,y0,px0,py0,delx0,dely0,delpx0,delpy0,delx1,dely1,delpx1,delpy1]))
    
y0 = 2.5
py0 = 0

delta = Ohm**2 *y0**2 - 2*(0.5*py0**2 + potential(x0,y0,0) - H0) 
px0 = -Ohm*y0 + np.sqrt(delta)
   
c = RK6(100,0.0003, np.array([x0,y0,px0,py0,delx0,dely0,delpx0,delpy0,delx1,dely1,delpx1,delpy1]))
end = time.time()
print(end-start)


plt.plot(b[0],b[1])
plt.plot(c[0],c[1])
plt.xscale("log")
plt.yscale("log")