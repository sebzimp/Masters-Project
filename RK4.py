# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 21:16:30 2022

@author: sebzi
"""

from numba import njit, cfunc, jit
from NumbaLSODA import lsoda_sig, lsoda

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
@njit(fastmath=True)
def Hz0(a): #a,t

    dx = a[2] + Ohm*a[1]
    
    dy =  a[3] - Ohm*a[0]
    
    dpx = Ohm*a[3] - a[0]*G* ( M1* (a[0]**2 + a[1]**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (a[0]**2 + a[1]**2/ qa**2 + (a2+ b2 )**2 )**(-1.5) )
    
    dpy =  -Ohm*a[2] - (a[1]/qa**2) *G* ( M1* (a[0]**2 + a[1]**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (a[0]**2 + a[1]**2/ qa**2 +  (a2 + b2 )**2)**(-1.5) )

    return  np.array([dx,dy,dpx,dpy]) 

@njit(fastmath=True)
def vect(x,y,px,py):
    v1 = px + Ohm*y
    v2 = py - Ohm*x
    v3 = Ohm*py- x*G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 + (a2+ b2 )**2 )**(-1.5) )
    v4 =  -Ohm*px- (y/qa**2) *G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 +  (a2 + b2 )**2)**(-1.5) )
    
    return [v1,v2,v3,v4]

#potential
@njit(fastmath=True)
def potential(x,y,z ):#,PARAMETERS = [a1,b1,M1,a2,b2,M2,qa,Ohm,G]):
    
    
    V = -G*M1*(x**2 + y**2/qa**2 + (a1 + (z**2/qb**2 + b1**2)**(0.5)  )**2)**(-0.5) - G*M2* (x**2 + y**2/qa**2 + (a2 + (z**2/qb**2 + b2**2)**(0.5)   )**2 )**(-0.5) 
    return V

#Hamiltonian
@njit(fastmath=True)
def Hamiltonian(x,y,px,py):
  
    H = 0.5*(px**2 + py**2) + potential(x,y,0) -Ohm*(x*py - y*px)
    return H

@njit(fastmath=True)
def vect2(x,y,px,py):
    v1 = -(px + Ohm*y)
    v2 = -(py - Ohm*x)
    v3 = -(Ohm*py- x*G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 + (a2+ b2 )**2 )**(-1.5) ) )
    v4 =  -(-Ohm*px- (y/qa**2) *G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 +  (a2 + b2 )**2)**(-1.5) ))
    
    return [v1,v2,v3,v4]

@njit(fastmath=True)
def RK4(time,h,a0):
    
    t = np.arange(0,time, h)
    sol = [a0]
    
    for i in range(len(t)):
        
        an = sol[i]
        
        k1 = Hz0(an)
        
        k2 = Hz0(an +k1*h/2)
        
        k3 = Hz0(an + k2*h/2)
        
        k4 = Hz0(an + k3*h)
        

        anext = an + (1/6*k1+1/3*k2+1/3*k3+1/6*k4)*h 
        
        sol.append(anext)
        
    return  sol
#Energy
A = 43950
H0 = -4.18*A #hamiltonian

#fixed coordinate value
#y0 =0 
x0 = 0
#grid on which LDs are calculated



#calculating the LDs
#@jit()
def LDs(xarray,yarray):
    LD = []
    LD2 = []
    xax = []
    yax = []
    
    T = 5

    for i in range(len(xarray)):
        for j in range(len(yarray)):
            
            
      #      x0 = points_x[i] #x coordinate initial position
      #      px0 = points_y[j] #px coordinate initial position
    
            y0 = xarray[i] #x coordinate initial position
            py0 = yarray[j]
            
     #       delta = Ohm**2 *x0**2 - 2*(0.5*px0**2 + potential(x0,y0,0) +Ohm*y0*px0 - H0) 
            delta = Ohm**2 *y0**2 - 2*(0.5*py0**2 + potential(x0,y0,0) - H0) 
            
            if delta >=0:
                
       #         py0 = Ohm*x0 + np.sqrt( delta ) #py coordinate initial position for the given energy
                
                px0 = -Ohm*y0 + np.sqrt(delta)
                
                xax.append(y0)
                yax.append(py0)
                
                
                u0 = np.array([x0,y0,px0,py0]) # Initial conditions
            
                          
                usol = RK4(T,0.0001,u0)
                
                
                    
                x = np.zeros(len(usol))
                y = np.zeros(len(usol))
                px = np.zeros(len(usol))
                py = np.zeros(len(usol))
    
        #        py8 = []
                
                for k in range(len(usol)):
         #           region = usol[k][0]**2 + usol[k][1]**2 #+ (usol[k][2]/209.64)**2 +(usol[k][3]/209.64)**2 
         #           if region <= 100**2:
                        x[k] = (usol[k][0])
                        y[k] = (usol[k][1])
                        px[k] = (usol[k][2])
                        py[k] = (usol[k][3])
                    
    
    
    
                v =   vect(x,y,px,py)#vect(np.array(x),np.array(y),np.array(px),np.array(py))
                
                #cacluating the LD with p-norm, p =0.5
                intermedLD = np.sum(0.0001*np.abs(v)**0.5, axis=1)
                LD.append(np.sum(intermedLD))
    
    
    
                #back LD
    
                
            
                          
                usol2 = RK4(-T,-0.0001,u0)
                
                    
                x2 = np.zeros(len(usol2))
                y2 = np.zeros(len(usol2))
                px2 = np.zeros(len(usol2))
                py2 = np.zeros(len(usol2))
                
    
                for k in range(len(usol2)):
            #        region = usol2[k][0]**2 + usol2[k][1]**2 #+ (usol2[k][2]/209.64)**2 +(usol2[k][3]/209.64)**2 
            #        if region <= 100**2:
                        x2[k] = (usol2[k][0])
                        y2[k] = (usol2[k][1])
                        px2[k] = (usol2[k][2])
                        py2[k] = (usol2[k][3])
    
    
    
                v2 = vect(np.array(x2),np.array(y2),np.array(px2),np.array(py2))            
    
                intermedLD2 = np.sum(0.0001*np.abs(v2)**0.5, axis=1)
                LD2.append(np.sum(intermedLD2))

    LDprop = np.add(LD,LD2)
    
    return [LDprop ,xax,yax]


ax1_min,ax1_max = [2.75, 4]
ax2_min,ax2_max = [-90,90]
N1, N2 = [50,50]

grid_parameters = [[ax1_min, ax1_max, N1],[ax2_min, ax2_max, N2]]


x_min, x_max, Nx = grid_parameters[0]
y_min, y_max, Ny = grid_parameters[1]
points_x = np.linspace(x_min, x_max, Nx)
points_y = np.linspace(y_min, y_max, Ny)      

'''          
y0 = 2.5
py0 = 0

delta = Ohm**2 *y0**2 - 2*(0.5*py0**2 + potential(x0,y0,0) - H0) 

px0 = -Ohm*y0 + np.sqrt(delta)           

b = RK4(5,0.0001, np.array([x0,y0,px0,py0]))

T = b[0]
sol = b[1]

error = []
for i in range(len(T)):
    
    re = np.abs( (H0- Hamiltonian(sol[i][0], sol[i][1],sol[i][2], sol[i][3]) )/H0    )
     
    error.append(re)         

end = time.time()
print(end-start)

plt.plot(T,error)'''
            
sol = LDs(points_x,points_y)


end = time.time()
print(end-start)

#plotting the LDs
plt.figure(dpi=200)
#yax = np.true_divide(yax,209.64 )
#plt.title("Region of radius 20")
plt.scatter(sol[1],sol[2],c=sol[0] ,cmap = "plasma", s = 0.5)       
plt.colorbar(label = "LD") 
plt.xlabel("y")
plt.ylabel("$p_y$")