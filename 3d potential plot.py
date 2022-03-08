# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:48:23 2022

@author: sebzi
"""

from NumbaLSODA import lsoda_sig, lsoda
from numba import njit, cfunc, jit
import numpy as np
import matplotlib.pyplot as plt
import time


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

def potential(x,y,z,PARAMETERS = [a1,b1,M1,a2,b2,M2,qa,Ohm,G]):
    
    
    V = -G*M1*(x**2 + y**2/qa**2 + (a1 + (z**2/qb**2 + b1**2)**(0.5)  )**2)**(-0.5) - G*M2* (x**2 + y**2/qa**2 + (a2 + (z**2/qb**2 + b2**2)**(0.5)   )**2 )**(-0.5) \
    -0.5*Ohm**2*(x**2+y**2)
    return V


x = np.linspace(-7.5, 7.5, 50)
y = np.linspace(-5,5, 50)

X, Y = np.meshgrid(x, y)
Z = potential(X, Y,0)

 
levels = [-300000, -250000, -200000, -190000, -184137, -180000,-178000,  -175500]
plt.figure()
plt.contour(X,Y, Z,levels, colors='black')
plt.xlabel("x")
plt.ylabel("y")

lagx = [-3.587, 0,0,3.587,0]
lagy = [0,-3.895,3.895,0,0]
plt.plot(lagx,lagy, ".r")

for i in range(len(Z)):
    for j in range(len(Z[0])):
        if Z[i][j] <-200000:
            Z[i][j] = -200000

fig = plt.figure()
ax = fig.gca(projection='3d')
ax1 = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, Z,cmap="plasma", linewidth=0,antialiased=True, rstride=1,cstride=1)

a = potential(np.array(lagx),np.array(lagy),0)
a = a[:-1]
a = np.append(a,-200000)
ax.scatter(lagx,lagy, a,s =20)
plt.xlabel("x")
plt.ylabel("y")
ax.view_init(30,190)
fig.colorbar(surf, shrink=0.5, aspect=10)
