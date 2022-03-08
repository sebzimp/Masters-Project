# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 14:47:31 2022

@author: sebzi
"""

from NumbaLSODA import lsoda_sig, lsoda
from numba import njit, cfunc, jit
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import optimize

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

def effectivex(x):
   
    y =0

    f =   x*G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 + (a2+ b2 )**2 )**(-1.5) ) - Ohm**2*x
         
    return f

def effectivey(y):
    x=0
    

    f =   y/qa*G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1 )**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 + (a2+ b2 )**2 )**(-1.5) ) - Ohm**2*y
         
    return f
x = np.linspace(-10,10, 501)


root1 = optimize.brentq(effectivex, -7.5, -2.5)

root2 = optimize.brentq(effectivex, 7.5, 2.5)

root3 = optimize.brentq(effectivey, -7.5, -2.5)

root4 = optimize.brentq(effectivey, 7.5, 2.5)