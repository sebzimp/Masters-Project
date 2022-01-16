# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 11:45:56 2021

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

#equation of motion
@cfunc(lsoda_sig)
def Hz0(t, a , da,p): #a,t

    da[0] = a[2] + Ohm*a[1]
    
    da[1] =  a[3] - Ohm*a[0]
    
    da[2] = Ohm*a[3] - a[0]*G* ( M1* (a[0]**2 + a[1]**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (a[0]**2 + a[1]**2/ qa**2 + (a2+ b2 )**2 )**(-1.5) )
    
    da[3] =  -Ohm*a[2] - (a[1]/qa**2) *G* ( M1* (a[0]**2 + a[1]**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (a[0]**2 + a[1]**2/ qa**2 +  (a2 + b2 )**2)**(-1.5) )


#EOM in backwards time
@cfunc(lsoda_sig)
def Hz1(t, a , da,p): #a,t

    da[0] = -(a[2] + Ohm*a[1])
    
    da[1] =  -(a[3] - Ohm*a[0])
    
    da[2] = -(Ohm*a[3] - a[0]*G* ( M1* (a[0]**2 + a[1]**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (a[0]**2 + a[1]**2/ qa**2 + (a2+ b2 )**2 )**(-1.5) ) )
    
    da[3] =  -(-Ohm*a[2] - (a[1]/qa**2) *G* ( M1* (a[0]**2 + a[1]**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (a[0]**2 + a[1]**2/ qa**2 +  (a2 + b2 )**2)**(-1.5) ) )


def vect(x,y,px,py):
    v1 = px + Ohm*y
    v2 = py - Ohm*x
    v3 = Ohm*py- x*G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 + (a2+ b2 )**2 )**(-1.5) )
    v4 =  -Ohm*px- (y/qa**2) *G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 +  (a2 + b2 )**2)**(-1.5) )
    
    return [v1,v2,v3,v4]

#potential
def potential(x,y,z,PARAMETERS = [a1,b1,M1,a2,b2,M2,qa,Ohm,G]):
    
    
    V = -G*M1*(x**2 + y**2/qa**2 + (a1 + (z**2/qb**2 + b1**2)**(0.5)  )**2)**(-0.5) - G*M2* (x**2 + y**2/qa**2 + (a2 + (z**2/qb**2 + b2**2)**(0.5)   )**2 )**(-0.5) 
    return V

#Hamiltonian
def Hamiltonian(x,y,px,py):
  
    H = 0.5*(px**2 + py**2) + potential(x,y,0) -Ohm*(x*py - y*px)
    return H
 

#Energy
A = 43950
H0 = -5.207*A #hamiltonian

#fixed coordinate value
y0 =0 

#grid on which LDs are calculated
ax1_min,ax1_max = [-1, 1]
ax2_min,ax2_max = [-500,500]
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
LD = []
xax = []
yax = []

#integration time
T = 5000 #timesteps
t = np.linspace(0.0, 5.0, T)

#calculating the LDs
for i in range(len(points_x)):
    for j in range(len(points_y)):
        
        
        x0 = points_x[i] #x coordinate initial position
        px0 = points_y[j] #px coordinate initial position
        
        delta = Ohm**2 *x0**2 - 2*(0.5*px0**2 + potential(x0,y0,0) - H0) 
        
        if delta >=0:
            
            py0 = Ohm*x0 + np.sqrt( delta ) #py coordinate initial position for the given energy
            
            xax.append(x0)
            yax.append(px0)
    
            funcptr = Hz0.address # address to ODE function for
            
  #          funcptr = Hz1.address #back
            
            u0 = np.array([x0,y0,px0,py0]) # Initial conditions
        
                      
            usol, success = lsoda(funcptr, u0, t,rtol = 1.0e-8, atol = 1.0e-9) #solving EoM
            
                
            x = []
            y = []
            px = []
            py = []
            
    
            
            for k in range(len(usol)):
                x.append(usol[k][0])
                y.append(usol[k][1])
                px.append(usol[k][2])
                py.append(usol[k][3])
    
            v = vect(-np.array(x),-np.array(y),-np.array(px),-np.array(py))
            
    
            #cacluating the LD with p-norm, p =0.5
            intermedLD = np.sum(0.001*np.abs(v)**1, axis=1)
            LD.append(np.sum(intermedLD))
                   
end = time.time()
print(end - start)


#plotting the LDs
plt.figure(dpi=200)
yax = np.true_divide(yax,209.64 )
plt.scatter(xax,yax,c=LD ,cmap = "plasma", s = 0.5)       
plt.colorbar(label = "LD") 
plt.xlabel("x")
plt.ylabel("$p_x$")
