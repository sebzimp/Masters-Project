# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 12:13:45 2022

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

@jit
def vect(x,y,px,py):
    v1 = px + Ohm*y
    v2 = py - Ohm*x
    v3 = Ohm*py- x*G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 + (a2+ b2 )**2 )**(-1.5) )
    v4 =  -Ohm*px- (y/qa**2) *G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 +  (a2 + b2 )**2)**(-1.5) )
    
    return [v1,v2,v3,v4]

#potential
@jit
def potential(x,y,z,PARAMETERS = [a1,b1,M1,a2,b2,M2,qa,Ohm,G]):
    
    
    V = -G*M1*(x**2 + y**2/qa**2 + (a1 + (z**2/qb**2 + b1**2)**(0.5)  )**2)**(-0.5) - G*M2* (x**2 + y**2/qa**2 + (a2 + (z**2/qb**2 + b2**2)**(0.5)   )**2 )**(-0.5) 
    return V

#Hamiltonian
@jit
def Hamiltonian(x,y,px,py):
  
    H = 0.5*(px**2 + py**2) + potential(x,y,0) -Ohm*(x*py - y*px)
    return H

@jit
def vect2(x,y,px,py):
    v1 = -(px + Ohm*y)
    v2 = -(py - Ohm*x)
    v3 = -(Ohm*py- x*G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 + (a2+ b2 )**2 )**(-1.5) ) )
    v4 =  -(-Ohm*px- (y/qa**2) *G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 +  (a2 + b2 )**2)**(-1.5) ))
    
    return [v1,v2,v3,v4]

#Energy
A = 43950
H0 = -4.18*A #hamiltonian

#fixed coordinate value
#y0 =0 
x0 = 0
#grid on which LDs are calculated
ax1_min,ax1_max = [2.75, 4]
ax2_min,ax2_max = [-90,90]
N1, N2 = [10,10]

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
LD2 = []
xax = []
yax = []


#integration time
T = 5000 #timesteps
t = np.linspace(0.0, 5.0, T)

#calculating the LDs

for i in range(len(points_x)):
    for j in range(len(points_y)):
        
        
  #      x0 = points_x[i] #x coordinate initial position
  #      px0 = points_y[j] #px coordinate initial position

        y0 = points_x[i] #x coordinate initial position
        py0 = points_y[j]
        
 #       delta = Ohm**2 *x0**2 - 2*(0.5*px0**2 + potential(x0,y0,0) +Ohm*y0*px0 - H0) 
        delta = Ohm**2 *y0**2 - 2*(0.5*py0**2 + potential(x0,y0,0) - H0) 
        
        if delta >=0:
            
   #         py0 = Ohm*x0 + np.sqrt( delta ) #py coordinate initial position for the given energy
            
            px0 = -Ohm*y0 + np.sqrt(delta)
            
            xax.append(y0)
            yax.append(py0)
    
            funcptr = Hz0.address # address to ODE function for
            
            
            u0 = np.array([x0,y0,px0,py0]) # Initial conditions
        
                      
            usol, success = lsoda(funcptr, u0, t,rtol = 1.0e-8, atol = 1.0e-9) #solving EoM
            
                
            x = []
            y = []
            px = []
            py = []

            
            for k in range(len(usol)):
     #           region = usol[k][0]**2 + usol[k][1]**2 #+ (usol[k][2]/209.64)**2 +(usol[k][3]/209.64)**2 
     #           if region <= 100**2:
                    x.append(usol[k][0])
                    y.append(usol[k][1])
                    px.append(usol[k][2])
                    py.append(usol[k][3])
                    '''
                    if t[k] <= 12 <t[k+1] :
                        v = vect(np.array(x),np.array(y),np.array(px),np.array(py))
                        
                
                        #cacluating the LD with p-norm, p =0.5
                        intermedLD = np.sum(0.001*np.abs(v)**0.5, axis=1)
                        LD6.append(np.sum(intermedLD))                    

                    if t[k] <= 15 <t[k+1] :
                        v = vect(np.array(x),np.array(y),np.array(px),np.array(py))
                        
                
                        #cacluating the LD with p-norm, p =0.5
                        intermedLD = np.sum(0.001*np.abs(v)**0.5, axis=1)
                        LD7.append(np.sum(intermedLD))   

                    if t[k] <= 18 <t[k+1] :
                        v = vect(np.array(x),np.array(y),np.array(px),np.array(py))
                        
                
                        #cacluating the LD with p-norm, p =0.5
                        intermedLD = np.sum(0.001*np.abs(v)**0.5, axis=1)
                        LD8.append(np.sum(intermedLD))                    

                    if t[k] <= 9 <t[k+1] :
                        v = vect(np.array(x),np.array(y),np.array(px),np.array(py))
                        
                
                        #cacluating the LD with p-norm, p =0.5
                        intermedLD = np.sum(0.001*np.abs(v)**0.5, axis=1)
                        LD9.append(np.sum(intermedLD))  '''
      #          if region  <= 10**2:
      #              x10.append(usol[k][0])
      #              y10.append(usol[k][1])
      #              px10.append(usol[k][2])
      #              py10.append(usol[k][3])
                
     #           if region  <= 8**2:
     #               x8.append(usol[k][0])
     #               y8.append(usol[k][1])
     #               px8.append(usol[k][2])
     #               py8.append(usol[k][3])                
     #           else:
     #               break
            


            v = vect(np.array(x),np.array(y),np.array(px),np.array(py))
            
    
            #cacluating the LD with p-norm, p =0.5
            intermedLD = np.sum(0.001*np.abs(v)**0.5, axis=1)
            LD.append(np.sum(intermedLD))


    #        v10 = vect(np.array(x10),np.array(y10),np.array(px10),np.array(py10))
            
    
            #cacluating the LD with p-norm, p =0.5
   #         intermedLD = np.sum(0.001*np.abs(v10)**0.5, axis=1)
   #         LD10for.append(np.sum(intermedLD))
            
   #         v8 = vect(np.array(x8),np.array(y8),np.array(px8),np.array(py8))
            
    
            #cacluating the LD with p-norm, p =0.5
   #         intermedLD = np.sum(0.001*np.abs(v8)**0.5, axis=1)
   #         LD8for.append(np.sum(intermedLD))
            
            #back LD
            funcptr2 = Hz1.address #back
            
        
                      
            usol2, success = lsoda(funcptr2, u0, t,rtol = 1.0e-8, atol = 1.0e-9) #solving EoM
            
                
            x2 = []
            y2 = []
            px2 = []
            py2 = []
            
      #      xb10 = []
      #      yb10 = []
      #      pxb10 = []
      #      pyb10 = []
            
      #      xb8 = []
      #      yb8 = []
      #      pxb8 = []
      #      pyb8 = []    
            
            for k in range(len(usol2)):
        #        region = usol2[k][0]**2 + usol2[k][1]**2 #+ (usol2[k][2]/209.64)**2 +(usol2[k][3]/209.64)**2 
        #        if region <= 100**2:
                    x2.append(usol2[k][0])
                    y2.append(usol2[k][1])
                    px2.append(usol2[k][2])
                    py2.append(usol2[k][3])

                    '''
                    if t[k] <= 6 <t[k+1] :
                        v = vect2(np.array(x2),np.array(y2),np.array(px2),np.array(py2))
                        
                
                        #cacluating the LD with p-norm, p =0.5
                        intermedLD = np.sum(0.001*np.abs(v)**0.5, axis=1)
                        LD6b.append(np.sum(intermedLD))                    

                    if t[k] <= 7 <t[k+1] :
                        v = vect2(np.array(x2),np.array(y2),np.array(px2),np.array(py2))
                        
                
                        #cacluating the LD with p-norm, p =0.5
                        intermedLD = np.sum(0.001*np.abs(v)**0.5, axis=1)
                        LD7b.append(np.sum(intermedLD))   

                    if t[k] <= 8 <t[k+1] :
                        v = vect2(np.array(x2),np.array(y2),np.array(px2),np.array(py2))
                        
                
                        #cacluating the LD with p-norm, p =0.5
                        intermedLD = np.sum(0.001*np.abs(v)**0.5, axis=1)
                        LD8b.append(np.sum(intermedLD))                    

                    if t[k] <= 9 <t[k+1] :
                        v = vect2(np.array(x2),np.array(y2),np.array(px2),np.array(py2))
                        
                
                        #cacluating the LD with p-norm, p =0.5
                        intermedLD = np.sum(0.001*np.abs(v)**0.5, axis=1)
                        LD9b.append(np.sum(intermedLD))      '''
                
     #           if region  <= 10**2:
     #               xb10.append(usol2[k][0])
     #               yb10.append(usol2[k][1])
     #               pxb10.append(usol2[k][2])
     #               pyb10.append(usol2[k][3])
                
     #           if region  <= 8**2:
     #               xb8.append(usol2[k][0])
     #               yb8.append(usol2[k][1])
     #               pxb8.append(usol2[k][2])
    #                pyb8.append(usol2[k][3])                
     #           else:
     #               break
            


            v2 = vect2(np.array(x2),np.array(y2),np.array(px2),np.array(py2))            

            intermedLD2 = np.sum(0.001*np.abs(v2)**0.5, axis=1)
            LD2.append(np.sum(intermedLD2))
                  

      #      vb10 = vect2(np.array(xb10),np.array(yb10),np.array(pxb10),np.array(pyb10))
            
    
            #cacluating the LD with p-norm, p =0.5
     #       intermedLD = np.sum(0.001*np.abs(vb10)**0.5, axis=1)
     #       LD10back.append(np.sum(intermedLD))
            
    #        vb8 = vect2(np.array(xb8),np.array(yb8),np.array(pxb8),np.array(pyb8))
            
    
            #cacluating the LD with p-norm, p =0.5
    #        intermedLD = np.sum(0.001*np.abs(vb8)**0.5, axis=1)
    #        LD8back.append(np.sum(intermedLD))            
            
            
end = time.time()
print(end - start)

LDprop = np.add(LD,LD2)


#plotting the LDs
plt.figure(dpi=200)
yax = np.true_divide(yax,209.64 )
#plt.title("Region of radius 20")
plt.scatter(xax,yax,c=LDprop ,cmap = "plasma", s = 0.5)       
plt.colorbar(label = "LD") 
plt.xlabel("y")
plt.ylabel("$p_y$")
plt.show()
'''
#LDprop10 = np.add(LD10for,LD10back)

LDprop6 = np.add(LD6,LD6b)
LDprop7 = np.add(LD7,LD7b)
LDprop8 = np.add(LD8,LD8b)
LDprop9 = np.add(LD9,LD9b)

#plotting the LDs
plt.figure(dpi=200)
plt.title("$\tau = 6$")
plt.scatter(xax,yax,c=LDprop6 ,cmap = "plasma", s = 0.5)       
plt.colorbar(label = "LD") 
plt.xlabel("y")
plt.ylabel("$p_y$")

plt.figure(dpi=200)
plt.title("$\tau = 7$")
plt.scatter(xax,yax,c=LDprop7 ,cmap = "plasma", s = 0.5)       
plt.colorbar(label = "LD") 
plt.xlabel("y")
plt.ylabel("$p_y$")

plt.figure(dpi=200)
plt.title("$\tau = 8$")
plt.scatter(xax,yax,c=LDprop8 ,cmap = "plasma", s = 0.5)       
plt.colorbar(label = "LD") 
plt.xlabel("y")
plt.ylabel("$p_y$")

plt.figure(dpi=200)
plt.title("$\tau = 9$")
plt.scatter(xax,yax,c=LDprop9 ,cmap = "plasma", s = 0.5)       
plt.colorbar(label = "LD") 
plt.xlabel("y")
plt.ylabel("$p_y$")
'''
#LDprop8 = np.add(LD8for,LD8back)


#plotting the LDs
#plt.figure(dpi=200)
#yax = np.true_divide(yax,209.64 )
#plt.title("Region of radius 8")
#plt.scatter(xax,yax,c=LDprop8 ,cmap = "plasma", s = 0.5)       
#plt.colorbar(label = "LD") 
#plt.xlabel("y")
#plt.ylabel("$p_y$")
