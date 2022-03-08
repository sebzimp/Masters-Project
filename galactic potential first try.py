# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 17:38:52 2021

@author: sebzi
"""

import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('D:/New folder/masters/github code/ldds-develop')

import ldds
from ldds.base import compute_lagrangian_descriptor, perturb_field
from ldds.tools import draw_all_lds, draw_ld
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import time

start = time.time()

a1 = 0
b1= 0.495
M1 = 2.05*10**10
a2 = 7.258
b2 = 0.520
M2 = 25.47*10**10
qa = 1.2
Ohm = 60
G = 4.3009*10**(-6)

def my_vector_field(t, u, PARAMETERS = [a1,b1,M1,a2,b2,M2,qa,Ohm,G]):
    """
    Returns 2-DoF vector field (Double Well system), for an array of points in phase space.
    Number of model parameters: 0 . PARAMETERS = [None]
    Functional form: v = (p_x, p_y, x - x**3, -y), with u = (x, y, p_x, p_y)
    Parameters
    ----------
    t : float
    fixed time-point of vector field, for all points in phase space.
    u : array_like, shape(n,)
    Points in phase space.
    PARAMETERS : list of floats
    Vector field parameters.
    Returns
    -------
    v : array_like, shape(n,)
    Vector field corresponding to points u, in phase space at time t.
    """

    
    N_dims = u.shape[-1]
    points_positions = u.T[:int(N_dims/2)]
    points_momenta = u.T[int(N_dims/2):]
    x, y = points_positions
    p_x, p_y = points_momenta
    # Hamiltonian Model Parameter
    # Vector field defintion
    v_x = p_x + Ohm*y
    v_y = p_y - Ohm*x
    v_p_x = Ohm*p_y - x*G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1)**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 + (a2 + b2)**2 )**(-1.5) )

    v_p_y = Ohm*p_x - (y/qa**2) *G* ( M1* (x**2 + y**2/ qa**2 + (a1 + b1)**2 )**(-1.5) + M2* (x**2 + y**2/ qa**2 + (a2 + b2)**2 )**(-1.5) )
    v = np.array([v_x, v_y, v_p_x, v_p_y]).T
    return v

def potential(positions, PARAMETERS = [a1,b1,M1,a2,b2,M2,qa,Ohm,G]):
    x, y = positions.T
    

    # Function parameters
    # None
    # Potential energy function
    V = -G*M1*(x**2 + y**2/qa**2 + (a1 + b1)**2 )**(-0.5) - G*M2* (x**2 + y**2/qa**2 + (a2 + b2)**2 )**(-0.5) - 1/2 * Ohm**2 *(x**2 + y**2)
    return V

def Hamiltonian(t,u):
    
    H = 0.5*(u[2]**2 + u[3]**2) + potential(u[0],u[1],0) -Ohm*(u[0]*u[3] - u[1]*u[2])
    return H


# Integration parameters
t0 = 0
tau = 10
# Lp-norm, p-value
p_value = 1/2
# Mesh visualisation slice parameters
A = 43950
H0 = -5.207 * A# Energy level
ax1_min,ax1_max = [0.18312784, 0.18312784]
ax2_min,ax2_max = [-0,0]
N1, N2 = [1, 1]
# Box escape condition
box_boundaries = False
# Miscellaneous grid parameters
dims_fixed = [0,1,0,0] # Variable ordering (x y p_x p_y)
dims_fixed_values = [0] # This can also be an array of values
dims_slice = [1,0,1,0] # Visualisation slice
momentum_sign = 1 # Direction of momentum that defines the slice - (1) positive / (-1) negative

box_boundaries = [[-10, 10], [-500, 500]]

potential_energy = potential
vector_field = my_vector_field
slice_parameters = [[ax1_min, ax1_max, N1],[ax2_min, ax2_max, N2]]
grid_parameters = {
        'slice_parameters' : slice_parameters,
        'dims_slice' : dims_slice,
        'dims_fixed' : dims_fixed,
        'dims_fixed_values' : dims_fixed_values,
  #      'momentum_sign' : momentum_sign,
        'Hamiltonian': Hamiltonian,
        'energy_level': H0,
        'remaining_coordinate_bounds' : [0,450]
    }
print(1)
LD_forward = compute_lagrangian_descriptor(grid_parameters, vector_field, tau, p_value)#, box_boundaries)
print(2)
LD_backward = compute_lagrangian_descriptor(grid_parameters, vector_field, -tau, p_value)#, box_boundaries)

end = time.time()

print(end-start)

#figs = draw_ld_pair(LD_forward, get_gradient_magnitude(LD_forward), grid_parameters, )
figs = draw_all_lds(LD_forward, LD_backward, grid_parameters, tau, p_value)