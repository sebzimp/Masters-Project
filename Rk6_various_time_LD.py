# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 12:39:31 2022

@author: sebzi
"""
import matplotlib.pyplot as plt
from numba import njit, cfunc, jit
#from NumbaLSODA import lsoda_sig, lsoda
#import numba
import numpy as np
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
#@jit()
#@cfunc("float64[::1](float64[::1])")
def Hz0(a): #a,t

    dx = a[2] + Ohm*a[1]
    
    dy =  a[3] - Ohm*a[0]
    
    dpx = Ohm*a[3] - a[0]*G* ( M1* (a[0]**2 + a[1]**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (a[0]**2 + a[1]**2/ qa**2 + (a2+ b2 )**2 )**(-1.5) )
    
    dpy =  -Ohm*a[2] - (a[1]/qa**2) *G* ( M1* (a[0]**2 + a[1]**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (a[0]**2 + a[1]**2/ qa**2 +  (a2 + b2 )**2)**(-1.5) )

    return  np.array([dx,dy,dpx,dpy]) 

@njit(fastmath=True)
#@jit()
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
#@jit()
#@cfunc("float64(float64, float64,float64)")
def potential(x,y,z ):#,PARAMETERS = [a1,b1,M1,a2,b2,M2,qa,Ohm,G]):
    
    
    V = -G*M1*(x**2 + y**2/qa**2 + (a1 + (z**2/qb**2 + b1**2)**(0.5)  )**2)**(-0.5) - G*M2* (x**2 + y**2/qa**2 + (a2 + (z**2/qb**2 + b2**2)**(0.5)   )**2 )**(-0.5) 
    return V

#Hamiltonian
@njit(fastmath=True)
#@jit()
#@cfunc("float64(float64, float64,float64, float64)")
def Hamiltonian(x,y,px,py):
  
    H = 0.5*(px**2 + py**2) + potential(x,y,0) -Ohm*(x*py - y*px)
    return H

@njit(fastmath=True)
#@jit()
#@cfunc("float64[::1](float64, float64,float64, float64)")
def vect2(x,y,px,py):
    v1 = -(px + Ohm*y)
    v2 = -(py - Ohm*x)
    v3 = -(Ohm*py- x*G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 + (a2+ b2 )**2 )**(-1.5) ) )
    v4 =  -(-Ohm*px- (y/qa**2) *G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 +  (a2 + b2 )**2)**(-1.5) ))
    
    return np.array([v1,v2,v3,v4])

@njit(fastmath=True)
#@jit()
#@cfunc("float64[:,::1](float64,float64, float64[::1])")
def RK6(time,h,a0):
    
    t = np.arange(0,time, h)
    n = len(t)
    
 #   sol = [a0]
 #   array = np.array([a0[0],a0[1],a0[2],a0[3]])
   # sol = np.array([[a0[0],a0[1],a0[2],a0[3]]])
    sol = np.zeros( (n+1, 4) , dtype=np.float64)
    sol[0] = a0
    for i in range(len(t)):
        
        an = sol[i]
        
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
        
      #  print(anext)
      #  print(sol)
        sol[i+1] = anext 
      #  print(sol)
 #       sol.append(anext)

  #  sol = np.array(sol)    
    return sol
#Energy
A = 43950
H0 = -4.18*A #hamiltonian

#fixed coordinate value
#y0 =0 

x0 = 0

res = 50
# grid on which LDs are calculated
ax1_min, ax1_max = [2.75, 4]
ax2_min, ax2_max = [-90, 90]
N1, N2 = [res, res]

grid_parameters = [[ax1_min, ax1_max, N1], [ax2_min, ax2_max, N2]]

x_min, x_max, Nx = grid_parameters[0]
y_min, y_max, Ny = grid_parameters[1]
points_x = np.linspace(x_min, x_max, Nx)
points_y = np.linspace(y_min, y_max, Ny)


#calculating the LDs
@njit(fastmath=True)
#@jit()
#@cfunc("float64[:,::1](float64[::1],float64[::1])")
def LDs(xrange):
    LD = []
    LD2 = []
    xax = []
    yax = []
    
    LD6 = []
    LD7 = []
    LD8 = []
    LD9 = []
    
    LD6b = []
    LD7b = []
    LD8b = []
    LD9b = []   
    
    T = 10.0

    for i in range(len(xrange)):
        for j in range(len(points_y)):
            
            
      #      x0 = points_x[i] #x coordinate initial position
      #      px0 = points_y[j] #px coordinate initial position
    
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
            
                          
                usol = RK6(T,0.0004,u0)
                
                t = np.arange(0,T, 0.0004)
                
    
        #        py8 = []
                forLD = 0
                for k in range(len(usol)):
         #           region = usol[k][0]**2 + usol[k][1]**2 #+ (usol[k][2]/209.64)**2 +(usol[k][3]/209.64)**2 
         #           if region <= 100**2:
                        v = vect(usol[k][0],usol[k][1],usol[k][2],usol[k][3])
                        
                        pnorm = np.abs(v[0])**0.5+np.abs(v[1])**0.5+np.abs(v[2])**0.5+np.abs(v[3])**0.5
                        
                        forLD = forLD + 0.0004*pnorm
                        
                        if t[k] <= 6 < t[k+1]:
                            LD6.append(forLD)

                        if t[k] <= 7 < t[k+1]:
                            LD7.append(forLD)                        

                        if t[k] <= 8 < t[k+1]:
                            LD8.append(forLD)

                        if t[k] <= 9 < t[k+1]:
                            LD9.append(forLD)       
                            
                LD.append(forLD)
    
    
    
                #back LD
    
                
            
                          
                usol2 = RK6(-T,-0.0004,u0)
                
                t2 = np.arange(0,-T, -0.0004)
                
                
                   
                backLD = 0
                
    
                for k in range(len(usol2)):
            #        region = usol2[k][0]**2 + usol2[k][1]**2 #+ (usol2[k][2]/209.64)**2 +(usol2[k][3]/209.64)**2 
            #        if region <= 100**2:
                        v2 = vect(usol2[k][0],usol2[k][1],usol2[k][2],usol2[k][3])
                        
                        pnorm2 = np.abs(v2[0])**0.5+np.abs(v2[1])**0.5+np.abs(v2[2])**0.5+np.abs(v2[3])**0.5
                        
                        backLD = backLD + 0.0004*pnorm2
                    

                        if t2[k] >= -6 > t2[k+1]:
                            LD6b.append(forLD)

                        if t2[k] >= -7 > t2[k+1]:
                            LD7b.append(forLD)
                            
                        if t2[k] >= -8 > t2[k+1]:
                            LD8b.append(forLD)

                        if t2[k] >= -9 > t2[k+1]:
                            LD9b.append(forLD)   
                            
                LD2.append(backLD)

#    LDprop = np.add(LD,LD2)
    
    return np.array([LD,LD2 ,LD6,LD6b, LD7,LD7b,LD8,LD8b,LD9,LD9b,])




import multiprocessing as mp

def main():
    pool = mp.Pool(mp.cpu_count())
    x_split=np.array_split(points_x,  mp.cpu_count())

    result = pool.map(LDs, x_split)


    forLD = np.array([])
    backLD = np.array([])

    forLD6 = np.array([])
    backLD6 = np.array([])

    forLD7 = np.array([])
    backLD7 = np.array([])    

    forLD8 = np.array([])
    backLD8 = np.array([])

    forLD9 = np.array([])
    backLD9 = np.array([])    
    
    for i in range(len(result)):
        forLD = np.concatenate((forLD,np.array(result[i][0])))
        backLD = np.concatenate((backLD,np.array(result[i][1])))

        forLD6 = np.concatenate((forLD6,np.array(result[i][2])))
        backLD6 = np.concatenate((backLD6,np.array(result[i][3])))

        forLD7 = np.concatenate((forLD7,np.array(result[i][4])))
        backLD7 = np.concatenate((backLD7,np.array(result[i][5])))

        forLD8 = np.concatenate((forLD8,np.array(result[i][6])))
        backLD8 = np.concatenate((backLD8,np.array(result[i][7])))

        forLD9 = np.concatenate((forLD9,np.array(result[i][8])))
        backLD9 = np.concatenate((backLD9,np.array(result[i][9])))

    '''
    np.savetxt("grid10000RK6tau10forLD.txt",forLD)
    np.savetxt("grid10000RK6tau10backLD.txt",backLD)

    np.savetxt("grid10000RK6tau6forLD.txt",forLD6)
    np.savetxt("grid10000RK6tau6backLD.txt",backLD6)

    np.savetxt("grid10000RK6tau7forLD.txt",forLD7)
    np.savetxt("grid10000RK6tau7backLD.txt",backLD7)

    np.savetxt("grid10000RK6tau8forLD.txt",forLD8)
    np.savetxt("grid10000RK6tau8backLD.txt",backLD8)

    np.savetxt("grid10000RK6tau9forLD.txt",forLD9)
    np.savetxt("grid10000RK6tau9backLD.txt",backLD9)
    '''
if __name__ == "__main__":
  main()
  end = time.perf_counter()
  print(end - start)
  plt.show()
