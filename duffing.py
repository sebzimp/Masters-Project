# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 22:22:57 2022

@author: sebzi
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
import time
from scipy.optimize import brentq

from NumbaLSODA import lsoda_sig, lsoda
from numba import njit, cfunc, jit
from scipy.integrate import solve_ivp


from scipy import integrate

start = time.time()

@cfunc(lsoda_sig)
def Hz0(t,a,da,p): #a,t


    da[0] = a[1]
    da[1] =  a[0] - a[0]**3


@cfunc(lsoda_sig)
def Hz1(t,a,da,p): #a,t


    da[0] = -a[1]
    da[1] =  -(a[0] - a[0]**3)


    
#ICs
def Hamiltonian(x,px):
  
    H = 0.5*(px**2 -x**2)
    return H

ax1_min,ax1_max = [-1.6, 1.6]
ax2_min,ax2_max = [-1,1]
N1, N2 = [301,301]

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
LD2 = []
T = 1000 #timesteps
t = np.linspace(0.0, 10.0, T)

LDforcheck = []
LDbackcheck = []
  
for i in range(len(points_x)):
    for j in range(len(points_y)):
        
        
        x0 = points_x[i]
        px0 = points_y[j]
        

        x_plot.append(x0)
        px_plot.append(px0)
        
        u0 = np.array([x0,px0])


        funcptr = Hz0.address
        
        
        usol, success = lsoda(funcptr, u0, t )#,rtol = 1.0e-8, atol = 1.0e-9)
        
        x = []
        px = []


        
        for k in range(len(usol)):
            x.append(usol[k][0])
            px.append(usol[k][1])
        
        v = [np.array(px),np.array(x)- np.array(x)**3]
         
        intermedLD = np.sum(0.01*np.abs(v)**1, axis=1)
        LD.append(np.sum(intermedLD))

        x = np.array(x)
        px = np.array(px)
        
    #    integ = integrate.trapezoid([np.abs(x)**0.5,np.abs(px)**0.5],[t,t]) #try solve using integrator
    #    LDforcheck.append(np.sum(integ))       
        
        funcptr2 = Hz1.address
        
        usol2, success2 = lsoda(funcptr2, u0, t) #,rtol = 1.0e-8, atol = 1.0e-9)
        
        x2 = []
        px2 = []


        
        for k in range(len(usol2)):
            x2.append(usol2[k][0])
            px2.append(usol2[k][1])
        
        v2 = [-np.array(px2),-np.array(x2)+ np.array(x2)**3]

        x2 = np.array(x2)
        px2 = np.array(px2)
        
        intermedLD2 = np.sum(0.01*np.abs(v2)**1, axis=1)
        LD2.append(np.sum(intermedLD2))
        
    #    integ2 = integrate.trapezoid([np.abs(x2)**0.5,np.abs(px2)**0.5],[t,t]) #try solve using integrator
    #    LDbackcheck.append(np.sum(integ2)  )        
   
propLD = np.add(LD,LD2)        

#checkLD = np.add(LDforcheck,LDbackcheck) 

#print(propLD)
#print(checkLD)

end = time.time()
print(end - start)

plt.figure(dpi=200)
plt.scatter(x_plot,px_plot,c=propLD ,cmap = "plasma", s = 0.5)       
plt.colorbar(label = "LD") 
plt.xlabel("$x$")
plt.ylabel("$y$")
