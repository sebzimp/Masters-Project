# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 16:12:16 2022

@author: sebzi
"""

import numpy as np
import matplotlib.pyplot as plt
import math

import time
from numba import njit

start = time.time()


@njit(fastmath=True)
def LA(a,tau):
    x, y, px, py = a
    
    eLA = [x + px*tau, y + py*tau, px, py]
    return eLA


@njit(fastmath=True)
def LB(b,tau):
    x, y, px, py = b
    
    eLB = [x, y, px - x*(1.0+2.0*y)*tau, py + (y**2.0 - x**2.0 - y)*tau]
    return eLB




@njit(fastmath=True)
def norm(a):
    x,y,px,py = a
    norm = np.sqrt( x**2.0 + y**2.0 + px**2.0 + py**2.0 )
    return norm


@njit(fastmath=True)
def dot(a,b):
    x,y,z,v = a
    r,s,t,u = b

    dot = x*r + s*y + z*t + v*u
    return dot

@njit(fastmath=True)
def div(a,b):
    x,y,w,z = a
    r = x/b
    s = y/b
    t = w/b
    u = z/b
    return [r,s,t,u]
#need 4 orthonormal deviation vectors 
@njit(fastmath=True)
def spec(tim, firs,sec,thir):
#    x0 = 0.0
#    y0 = -0.25
#    py0 = 0.0
    x0 =firs
    y0 = sec
    py0 = thir
    
    
    H = 1.0/3.0 #hamiltonian
    
    px0 = np.sqrt(2.0*(H - 0.5*y0**2.0 - 0.5*py0**2.0 + 1.0/3.0 * y0**3.0)) 
    
    
        
    a0 = [x0,y0,px0,py0]
    
    #tim = 100000.0
    T = abs(tim)*10 #timesteps
    t = np.linspace(0.0, tim, T)
    
    
    if tim>0:
        tau = 0.1
    else:
        tau = -0.1
    
    ABA864 = [[x0], [y0], [px0] ,[py0]]
    
    a1 = 0.0711334264982231177779387300061549964174
    a2 = 0.241153427956640098736487795326289649618
    a3 = 0.521411761772814789212136078067994229991
    a4 = -0.333698616227678005726562603400438876027
    b1 = 0.183083687472197221961703757166430291072
    b2 = 0.310782859898574869507522291054262796375
    b3 = -0.0265646185119588006972121379164987592663
    b4 = 0.0653961422823734184559721793911134363710
    
    

    for i in range(0,4):
        ABA864[i].append(LA( LB( LA( LB(LA(LB(LA(LB(LA(LB(LA(LB(LA(LB (LA(a0,a1*tau) ,tau*b1) ,tau*a2) ,tau*b2 )  ,tau*a3),tau*b3), tau*a4),
                            tau*b4), tau*a4), tau*b3), tau*a3), tau*b2), tau*a2), tau*b1), tau*a1)[i])#first leapfrog
    
       
    a = [ ABA864[0][1], ABA864[1][1], ABA864[2][1], ABA864[3][1]]
    
    
    
    
    for i in range(1,len(t)):
        for j in range(0,4):
            ABA864[j].append(LA( LB( LA( LB(LA(LB(LA(LB(LA(LB(LA(LB(LA(LB (LA(a,a1*tau) ,tau*b1) ,tau*a2) ,tau*b2 )  ,tau*a3),tau*b3), tau*a4), #gives solution for all vlaues
                            tau*b4), tau*a4), tau*b3), tau*a3), tau*b2), tau*a2), tau*b1), tau*a1)[j])

        if ABA864[0][i+1]**2 + ABA864[1][i+1]**2 + ABA864[2][i+1]**2+ ABA864[3][i+1]**2 <= 15**2:
            a = [ ABA864[0][i+1], ABA864[1][i+1], ABA864[2][i+1], ABA864[3][i+1]]
        
        else:
            break
        
    return ABA864
    

#@njit(fastmath=True) 
def LD(tau, x0,y0,py0,p):
    a = spec(tau,x0,y0,py0)
    v = [np.array(a[2]) ,np.array(a[3]), -np.array(a[0]) - 2*np.array(a[0])*np.array(a[1]), -np.array(a[1])-np.array(a[0])**2 +np.array(a[1])**2]
    
    intermedLD = np.sum(0.1*np.abs(v)**p, axis=1)
    LD = np.sum(intermedLD)
    
    return LD

def LD2(tau, x0,y0,py0,p):
    a = spec(tau,x0,y0,py0)
    v = [-np.array(a[2]) ,-np.array(a[3]), np.array(a[0]) + 2*np.array(a[0])*np.array(a[1]), np.array(a[1])+np.array(a[0])**2 -np.array(a[1])**2]
    
    intermedLD = np.sum(0.1*np.abs(v)**p, axis=1)
    LD = np.sum(intermedLD)
    
    return LD
 
#@njit(fastmath=True) 
def various(y0,yL,stepy, py0, pyL, steppy):
    ylin = np.linspace(y0,yL, stepy)
    
    pylin = np.linspace(py0, pyL, steppy)
    
    M = []
    M2 = []
    yplot = []
    pyplot = []
    for i in range(len(ylin)):
        for j in range(len(pylin)):
            
            px2 = 2.0*(1.0/3.0 - 0.5*ylin[i]**2.0 - 0.5*pylin[j]**2.0 + 1.0/3.0 * ylin[i]**3.0)
            
            if px2 >= 0:
                yplot.append(ylin[i])
                pyplot.append(pylin[j])
                M.append(LD(10, 0.0, ylin[i], pylin[j], 0.5 ) )
                M2.append(LD(-10, 0.0, ylin[i], pylin[j], 0.5 ) )
    
    return yplot, pyplot,M, M2

a = various(-0.75,1.5, 500, -1,1, 500)

    
end = time.time()

print(end-start)

plt.figure(dpi=200)
plt.scatter(a[0],a[1],c=a[2] ,cmap = "plasma", s = 0.5)       
plt.colorbar(label = "LD") 
plt.xlabel("$x$")
plt.ylabel("$y$")

plt.figure(dpi=200)
plt.scatter(a[0],a[1],c=a[3] ,cmap = "plasma", s = 0.5)       
plt.colorbar(label = "LD") 
plt.xlabel("$x$")
plt.ylabel("$y$")

LagranDes = np.add(a[2],a[3])

plt.figure(dpi=200)
plt.scatter(a[0],a[1],c=LagranDes ,cmap = "plasma", s = 0.5)       
plt.colorbar(label = "LD") 
plt.xlabel("$y$")
plt.ylabel("$p_y$")