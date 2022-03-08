# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 13:13:18 2022

@author: sebzi
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import math
from numpy import *
import cmath
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import time
from scipy.optimize import brentq

#from numba import jit
start = time.time()

#parameter values
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

#equations of motion
#@jit
def Hz0(a,t):
    x, y,  px, py = a
    

    dadt = [px + Ohm*y, py - Ohm*x, Ohm*py - x*G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 + (a2+ b2 )**2 )**(-1.5) ),
           -Ohm*px - (y/qa**2) *G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 +  (a2 + b2 )**2)**(-1.5) )]
    return dadt

#y derivatives to find the PSS
#@jit
def ydevz0(b,t):
    x, y,  px, py, t = b
    

    dbdy = [ (px + Ohm*y)/(py - Ohm*x) , 1, (Ohm*py - x*G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 + (a2+ b2 )**2 )**(-1.5) ) )/ (py - Ohm*x),
           (-Ohm*px - (y/qa**2) *G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 +  (a2 + b2 )**2)**(-1.5) ) )/(py - Ohm*x), 1/(py - Ohm*x)]
    return dbdy

def xdevz0(b,t):
    x, y,  px, py, t = b
    

    dbdy = [ 1 , (py - Ohm*x )/(px + Ohm*y), (Ohm*py - x*G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 + (a2+ b2 )**2 )**(-1.5) ) )/ (px + Ohm*y),
           (-Ohm*px - (y/qa**2) *G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 +  (a2 + b2 )**2)**(-1.5) ) )/(px + Ohm*y), 1/(px + Ohm*y)]
    return dbdy

#potential
def potential(x,y,z,PARAMETERS = [a1,b1,M1,a2,b2,M2,qa,Ohm,G]):

    
    V = -G*M1*(x**2 + y**2/qa**2 + (a1 + (z**2/qb**2 + b1**2)**(0.5)  )**2)**(-0.5) - G*M2* (x**2 + y**2/qa**2 + (a2 + (z**2/qb**2 + b2**2)**(0.5)   )**2 )**(-0.5) 
    return V


#Hamiltonian
def Hamiltonian(x,y,px,py):
  
    H = 0.5*(px**2 + py**2) + potential(x,y,0) -Ohm*(x*py - y*px)
    return H


#coordinate value which is being kept fixed
 #keep 0

#energy
A = 43950
H0 = -4.1 *A #hamiltonian

#time for which the trahectory is calculated
T = 150000 #timesteps
t = np.linspace(0.0, 15.0, T)




def poincy(y0,py0):
    
    x0 = 0 #fixed coordinate value
    
    delta = Ohm**2 *y0**2 - 2*(0.5*py0**2 + potential(x0,y0,0) - H0) 
    
    if delta>= 0:
    
        px0 = -Ohm*y0 + np.sqrt(delta) #finding the required py IC for the given energy
        
        a0 = [x0, y0,px0, py0] #initial condition
        
    
        
        sol = odeint(Hz0, a0, t) #solving the eqns
        
        x = sol[:,0]
        y = sol[:,1]
          #     z = sol[:,2]
        px = sol[:,2]
        py = sol[:,3]
    
    return x,y,px,py

a = poincy(2.7,0)

x= a[0]
y = a[1]
py = a[3]

plt.figure()
plt.plot(y,py)

plt.figure()
plt.plot(x,y)