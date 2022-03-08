# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 14:53:21 2022

@author: sebzi
"""

from numba import njit, cfunc, jit
#from NumbaLSODA import lsoda_sig, lsoda
#import numba
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.perf_counter()

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
@njit(fastmath=True)
#@cfunc("float64[::1](float64[::1])")
def Hz0(a): #a,t

    dx = a[2] + Ohm*a[1]
    
    dy =  a[3] - Ohm*a[0]
    
    dpx = Ohm*a[3] - a[0]*G* ( M1* (a[0]**2 + a[1]**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (a[0]**2 + a[1]**2/ qa**2 + (a2+ b2 )**2 )**(-1.5) )
    
    dpy =  -Ohm*a[2] - (a[1]/qa**2) *G* ( M1* (a[0]**2 + a[1]**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (a[0]**2 + a[1]**2/ qa**2 +  (a2 + b2 )**2)**(-1.5) )

    return  np.array([dx,dy,dpx,dpy]) 

@njit(fastmath=True)
#@cfunc("float64[::1](float64, float64,float64, float64)")
def vect(x,y,px,py):
    v1 = px + Ohm*y
    v2 = py - Ohm*x
    v3 = Ohm*py- x*G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 + (a2+ b2 )**2 )**(-1.5) )
    v4 =  -Ohm*px- (y/qa**2) *G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 +  (a2 + b2 )**2)**(-1.5) )
    
    varr = np.array([v1,v2,v3,v4])
    return varr

#potential
@njit(fastmath=True)
#@cfunc("float64(float64, float64,float64)")
def potential(x,y,z ):#,PARAMETERS = [a1,b1,M1,a2,b2,M2,qa,Ohm,G]):
    
    
    V = -G*M1*(x**2 + y**2/qa**2 + (a1 + (z**2/qb**2 + b1**2)**(0.5)  )**2)**(-0.5) - G*M2* (x**2 + y**2/qa**2 + (a2 + (z**2/qb**2 + b2**2)**(0.5)   )**2 )**(-0.5) 
    return V

#Hamiltonian
@njit(fastmath=True)
#@cfunc("float64(float64, float64,float64, float64)")
def Hamiltonian(x,y,px,py):
  
    H = 0.5*(px**2 + py**2) + potential(x,y,0) -Ohm*(x*py - y*px)
    return H

@njit(fastmath=True)
#@cfunc("float64[::1](float64, float64,float64, float64)")
def vect2(x,y,px,py):
    v1 = -(px + Ohm*y)
    v2 = -(py - Ohm*x)
    v3 = -(Ohm*py- x*G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 + (a2+ b2 )**2 )**(-1.5) ) )
    v4 =  -(-Ohm*px- (y/qa**2) *G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 +  (a2 + b2 )**2)**(-1.5) ))
    
    return np.array([v1,v2,v3,v4])

@njit(fastmath=True)
#@cfunc("float64[:,::1](float64,float64, float64[::1])")
def EscapeT(time,h,a0):
    
    t = np.arange(0,time, h)
 
    an = a0
    escapeT = -1
    for i in range(len(t)):
        
        if abs(an[1])>= 4:
            escapeT = t[i]
            break
        
        else:

        
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
            
            an= anext 
 
    return escapeT

A = 43950
H0 = -4.18*A 

x0 = 0

res = 600
# grid on which LDs are calculated
ax1_min, ax1_max = [2.75, 3.6]
ax2_min, ax2_max = [-600, 600]
N1, N2 = [res, res]

grid_parameters = [[ax1_min, ax1_max, N1], [ax2_min, ax2_max, N2]]

x_min, x_max, Nx = grid_parameters[0]
y_min, y_max, Ny = grid_parameters[1]
points_x = np.linspace(x_min, x_max, Nx)
points_y = np.linspace(y_min, y_max, Ny)


@njit(fastmath=True)
def escapes(xrange):
    escape_times = []

    xax = []
    yax = []
    
    T = 200.0

    for i in range(len(xrange)):
        for j in range(len(points_y)):
            
    
            y0 = xrange[i] #x coordinate initial position
            py0 = points_y[j]
            
     #       delta = Ohm**2 *x0**2 - 2*(0.5*px0**2 + potential(x0,y0,0) +Ohm*y0*px0 - H0) 
            delta = Ohm**2 *y0**2 - 2*(0.5*py0**2 + potential(x0,y0,0) - H0) 
            
            if delta >=0:
                
       #         py0 = Ohm*x0 + np.sqrt( delta ) #py coordinate initial position for the given energy
                
                px0 = -Ohm*y0 + np.sqrt(delta)
                
                xax.append(y0)
                yax.append(py0)
                                
                u0 = np.array([x0,y0,px0,py0]) # Initial conditions
            
                          
                usol = EscapeT(T,0.0004,u0)
                
                escape_times.append(usol)
                
                
    

    


#    LDprop = np.add(LD,LD2)
    
    return escape_times


import multiprocessing as mp

def main():
    pool = mp.Pool(mp.cpu_count())
    x_split=np.array_split(points_x,  mp.cpu_count())

    result = pool.map(escapes, x_split)


    escape = np.array([])


    for i in range(len(result)):
        escape = np.concatenate((escape,np.array(result[i])))

    
    np.savetxt("E4.18grid600escapetimes.txt",escape)




if __name__ == "__main__":
  main()
  end = time.perf_counter()
  print(end - start)
  plt.show()