# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 13:28:25 2021

@author: sebzi
"""

from NumbaLSODA import lsoda_sig, lsoda
from numba import njit, cfunc, jit
import numpy as np
import matplotlib.pyplot as plt
import time

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

def potential(x,y,z,PARAMETERS = [a1,b1,M1,a2,b2,M2,qa,Ohm,G]):
    
    
    V = -G*M1*(x**2 + y**2/qa**2 + (a1 + (z**2/qb**2 + b1**2)**(0.5)  )**2)**(-0.5) - G*M2* (x**2 + y**2/qa**2 + (a2 + (z**2/qb**2 + b2**2)**(0.5)   )**2 )**(-0.5) 
    return V
#ICs

def Hamiltonian(x,y,px,py):
  
    H = 0.5*(px**2 + py**2) + potential(x,y,0) -Ohm*(x*py - y*px)
    return H

y0 = 0

A = 43950
H0 = -5.207*A

ax1_min,ax1_max = [-2, 2]
ax2_min,ax2_max = [-500,500]
N1, N2 = [100,500]

grid_parameters = [[ax1_min, ax1_max, N1],[ax2_min, ax2_max, N2]]


x_min, x_max, Nx = grid_parameters[0]
y_min, y_max, Ny = grid_parameters[1]
points_x = np.linspace(x_min, x_max, Nx)
points_y = np.linspace(y_min, y_max, Ny)
X, Y = np.meshgrid(points_x, points_y)  # Grid in phase space
# 2D grid + a zero column for LDs
mesh = np.transpose([X.ravel(), Y.ravel(), np.zeros(Nx*Ny)])
mask = False

real = []
x = []
px = []

for i in range(len(points_x)):
    for j in range(len(points_y)):
        
        
        x0 = points_x[i]
        px0 = points_y[j]
        
        x.append(x0)
        px.append(px0)
        
        delta = Ohm**2 *x0**2 - 2*(0.5*px0**2 + potential(x0,y0,0) - H0) 
        
        if delta >= 0:
            real.append(1)
        else:
            real.append(0)
        
plt.scatter(x,px,c=real ,cmap = "plasma", s = 0.5)       
plt.colorbar()        
        
        