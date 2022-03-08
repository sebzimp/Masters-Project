# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 16:00:07 2021

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

from numba import jit

from scipy.integrate import solve_ivp


start = time.time()

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

@jit
def Hz0(a,t): #a,t
    x, y, px, py = a

    dadt = [px + Ohm*y, py - Ohm*x, Ohm*py - x*G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 + (a2+ b2 )**2 )**(-1.5) ),
           -Ohm*px - (y/qa**2) *G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 +  (a2 + b2 )**2)**(-1.5) )]
    return dadt


def potential(x,y,z,PARAMETERS = [a1,b1,M1,a2,b2,M2,qa,Ohm,G]):
    
    
    V = -G*M1*(x**2 + y**2/qa**2 + (a1 + (z**2/qb**2 + b1**2)**(0.5)  )**2)**(-0.5) - G*M2* (x**2 + y**2/qa**2 + (a2 + (z**2/qb**2 + b2**2)**(0.5)   )**2 )**(-0.5) 
    return V
#ICs
def Hamiltonian(x,y,px,py):
  
    H = 0.5*(px**2 + py**2) + potential(x,y,0) -Ohm*(x*py - y*px)
    return H
 #keep 0

A = 43950
H0 = -5.207*A #hamiltonian


#create grid of initial conditions

ax1_min,ax1_max = [0.0, 0.75]
ax2_min,ax2_max = [0,0.5]
N1, N2 = [2,2]

grid_parameters = [[ax1_min, ax1_max, N1],[ax2_min, ax2_max, N2]]


x_min, x_max, Nx = grid_parameters[0]
y_min, y_max, Ny = grid_parameters[1]
points_x = np.linspace(x_min, x_max, Nx)
points_y = np.linspace(y_min, y_max, Ny)
X, Y = np.meshgrid(points_x, points_y)  # Grid in phase space
# 2D grid + a zero column for LDs
mesh = np.transpose([X.ravel(), Y.ravel(), np.zeros(Nx*Ny)])
mask = False



y0 =0 #fixed dimension

x_plot = []
px_plot = []
LD = []
for i in range(len(points_x)):
    for j in range(len(points_y)):
        
        
        x0 = points_x[i]
        px0 = points_y[j]
        
        py0 = Ohm*x0 + np.sqrt( Ohm**2 *x0**2 - 2*(0.5*px0**2 + potential(x0,y0,0) - H0)  )
        
        
   #     mid2 = time.time()
   #     print(mid2 - mid1)        
        if py0 >= 0:
            a0 = [x0, y0,px0, py0]
            x_plot.append(x0)
            px_plot.append(px0)
            
            T = 1000 #timesteps
            t = np.linspace(0.0, 10.0, T)

            mid1 = time.time()
    
            sol = odeint(Hz0, a0, t) #solving the eqns
           
#            sol = solve_ivp(Hz0, [0,10], a0, t_eval=t, rtol=1.0e-4, atol=1.0e-12, method='LSODA')
            
            mid2 = time.time()
            print(mid2 - mid1)  
            x = sol[:,0]
            y = sol[:,1]
              #     z = sol[:,2]
            px = sol[:,2]
            py = sol[:,3]
            
#            x = sol.y[0]
 #           y = sol.y[1]
 #           px = sol.y[2]
  #          py = sol.y[3]
               #     pz = sol[:,5]
            
            v = [x,y,px,py]
            
            mid3 = time.time()
            
            intermedLD = np.sum(np.abs(v)**0.5, axis=1)
            LD.append(np.sum(intermedLD))
            
            mid4 = time.time()
            print(mid4 - mid3) 
end = time.time()
print(end - start)
h = []
for i in range(len(t)):
    h.append(abs( (H0-Hamiltonian(x[i],y[i],px[i],py[i] ) )/H0))
plt.plot(t,h)
#plt.figure(dpi = 200)
#plt.scatter(x_plot,px_plot,c=LD ,cmap = "plasma", s = 0.5)







