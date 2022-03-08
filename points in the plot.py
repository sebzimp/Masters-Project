# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 21:50:27 2022

@author: sebzi
"""

from NumbaLSODA import lsoda_sig, lsoda
from numba import njit, cfunc, jit
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


yLan = 3.895332760636627
px = -Ohm*yLan

H0 = Hamiltonian(0,yLan,px,0)
print(H0)

xLan = 3.587196093186396
py = Ohm*xLan

H0 = Hamiltonian(xLan,0,0,py)
print(H0)

H0 = -4.18* 43950

#y0 =0 
x0 = 0
#grid on which LDs are calculated
ax1_min,ax1_max = [-10, 10]
ax2_min,ax2_max = [-700,700]
N1, N2 = [200,200]

grid_parameters = [[ax1_min, ax1_max, N1],[ax2_min, ax2_max, N2]]


x_min, x_max, Nx = grid_parameters[0]
y_min, y_max, Ny = grid_parameters[1]
points_x = np.linspace(x_min, x_max, Nx)
points_y = np.linspace(y_min, y_max, Ny)
X, Y = np.meshgrid(points_x, points_y)  # Grid in phase space
# 2D grid + a zero column for LDs
mesh = np.transpose([X.ravel(), Y.ravel(), np.zeros(Nx*Ny)])
mask = False

#empty arrays for plots
inplot = []
xax = []
yax = []

for i in range(len(points_x)):
    for j in range(len(points_y)):
        
        
        y0 = points_x[i] #x coordinate initial position
    #    px0 = points_y[j] #px coordinate initial position
        py0 = points_y[j]    
    
  #      delta = Ohm**2 *x0**2 - 2*(0.5*px0**2 + potential(x0,y0,0) - H0) 
        delta = Ohm**2 *y0**2 - 2*(0.5*py0**2 + potential(x0,y0,0) - H0) 
        xax.append(y0)
        yax.append(py0)
        if delta >=0:
            inplot.append(1)
            
        else:
            inplot.append(0)
plt.figure()           
plt.scatter(xax,yax,c=inplot ,cmap = "plasma", s = 0.5)  