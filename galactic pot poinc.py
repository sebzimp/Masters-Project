# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 12:23:06 2021

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
H0 = -4.18 *A #hamiltonian

#time for which the trahectory is calculated
T = 1000000 #timesteps
t = np.linspace(0.0, 100.0, T)



#function to calculate the PSS for a given initial x value (px is initally 0)
def poinc(x0):
    y0 = 0.0 
    px0 = 0
    
    delta = Ohm**2 *x0**2 - 2*(0.5*px0**2 + potential(x0,y0,0) - H0) 
    
    py0 = Ohm*x0 + np.sqrt( delta ) #finding the required py IC for the given energy
    
    a0 = [x0, y0,px0, py0] #initial condition
    
    xpoinc = [x0]
    pxpoinc = [px0]

    
    sol = odeint(Hz0, a0, t) #solving the eqns
    
    x = sol[:,0]
    y = sol[:,1]
      #     z = sol[:,2]
    px = sol[:,2]
    py = sol[:,3]
       #     pz = sol[:,5]
    
    
    #finding the intersections with the PSS
    for i in range(T-1):
        if y[i]<0 and y[i+1]>0  and py[i] -Ohm*x[i]>=0:
    
       #ics
                 ti = t[i]
                 yi = y[i]
                 pxi = px[i]
                 pyi = py[i]
   
                 xi = x[i]

    
     #              b0 = [xi, yi, zi, pxi, pyi, pzi,1.0/pyi]
                 b0 = [xi,yi,pxi,pyi, 1.0/pyi]  
                 Y = np.linspace(yi, 0.0 , 2) #from x point to 0 in one step

    
                 poinc = odeint(ydevz0, b0, Y)
    
                 xpoinc.append( poinc[:,0][1])
                 pxpoinc.append(  poinc[:,2][1])
                 
   #              xdot.append(poinc[:,2][1] + Ohm*poinc[:,1][1])
        
    return xpoinc, pxpoinc

def poincy(y0,py0):
    
    x0 = 0 #fixed coordinate value
    
    delta = Ohm**2 *y0**2 - 2*(0.5*py0**2 + potential(x0,y0,0) - H0) 
    
    if delta>= 0:
    
        px0 = -Ohm*y0 + np.sqrt(delta) #finding the required py IC for the given energy
        
        a0 = [x0, y0,px0, py0] #initial condition
        
        ypoinc = [y0]
        pypoinc = [py0]
    
        
        sol = odeint(Hz0, a0, t) #solving the eqns
        
        x = sol[:,0]
        y = sol[:,1]
          #     z = sol[:,2]
        px = sol[:,2]
        py = sol[:,3]
           #     pz = sol[:,5]
        
        
        #finding the intersections with the PSS
        for i in range(T-1):
            if x[i]<0 and x[i+1]>0  and px[i] +Ohm*y[i]>=0:
        
           #ics
                     ti = t[i]
                     yi = y[i]
                     pxi = px[i]
                     pyi = py[i]
       
                     xi = x[i]
    
        
         #              b0 = [xi, yi, zi, pxi, pyi, pzi,1.0/pyi]
                     b0 = [xi,yi,pxi,pyi, 1.0/pyi]  
                     X = np.linspace(xi, 0.0 , 2) #from x point to 0 in one step
    
        
                     poinc = odeint(xdevz0, b0, X)
        
                     ypoinc.append( poinc[:,1][1])
                     pypoinc.append(  poinc[:,3][1])
                     
       #              xdot.append(poinc[:,2][1] + Ohm*poinc[:,1][1])
        
        return ypoinc, pypoinc
    else:
        ypoinc = []
        pypoinc = []
        return ypoinc, pypoinc

h = []
#for i in range(T): #checking error
 #    h.append(abs( (H0- ( 0.5*(px[i]**2.0 + py[i]**2.0) + potential(x[i],y[i],0) -Ohm*(x[i]*py[i] - y[i]*px[i]) ))  /H0))
 #    j.append(abs(I - (x[i]*py[i] - y[i]*px[i])) /abs(I))


#plt.figure()
#plt.scatter(t,h, s=0.5)

#plt.figure()

xplots = []
yplots = []

p1 = 0.18312784
p2 = -0.59595941

#points = [p1, p1+0.1,p1+0.2,p1+0.3,p1+0.4, p1 +0.5, p1+0.6, p1+0.7, p2, p2+0.1, p2+0.2, p2+0.3,p2+0.4, -0.26] #intitial x values from the paper

#points = np.linspace(0,2,10)
#for i in range(len(points)):
#    a = poinc(points[i])
    
 #   x = a[0]
 #   px = a[1]
    
#    for j in range(len(x)):
#        xplots.append(x[j])
#        yplots.append(px[j])
#plt.xlabel('x')
#plt.ylabel('px')

#ypoints = np.linspace(-3.3,3.3,15)
#pypoints = np.linspace(-600,600, 15)

#for i in range(len(ypoints)):
#    for j in range(len(pypoints)):
#        a = poincy(ypoints[i],pypoints[j])
        
#        y = a[0]
#        py = a[1]
        
#        for k in range(len(y)):
#            xplots.append(y[k])
#            yplots.append(py[k])

a = poincy(3.6,0)

xplots = a[0]
yplots = a[1]



plt.figure()
yplots = np.true_divide(yplots,209.64 )
plt.xlabel('y')
plt.ylabel('$p_y$')
end = time.time()
print(end - start)
plt.scatter(xplots,yplots, s =0.5)
plt.show()