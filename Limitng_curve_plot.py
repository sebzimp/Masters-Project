# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 11:36:00 2022

@author: sebzi
"""

import numpy as np
from sympy import plot_implicit, Eq, symbols

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

def potential(x,y,PARAMETERS = [a1,b1,M1,a2,b2,M2,qa,Ohm,G]):
    
    
    V = -G*M1*(x**2 + y**2/qa**2 + (a1 + b1  )**2)**(-0.5) - G*M2* (x**2 + y**2/qa**2 + (a2 + b2)**2 )**(-0.5) 
    return V


def limiting_curve(y,py):
    LC = 0.5*py**2 + potential(0,y) - 0.5*Ohm**2*y**2
    
    return LC
    
y, py = symbols('y $p_y$')

def f(x,y):
    return x**2 +y**2

a, b = symbols('a b')
A = 43950
H0 = -4.1*A 

p1 = plot_implicit(Eq(limiting_curve(y,py), H0), (y,-4,4), (py,-600,600))

