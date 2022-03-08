# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:24:16 2021

@author: sebzi
"""

import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('D:/New folder/masters/github code/ldds-develop')

import ldds
from ldds.base import compute_lagrangian_descriptor, perturb_field
from ldds.tools import draw_all_lds, draw_ld
from ldds.vector_fields import Duffing1D, forcing
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import time

start = time.time()

def my_vector_field(t, u, PARAMETERS = [None]):
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
    v_x = p_x
    v_y = p_y
    v_p_x = x - x**3
    v_p_y = -y
    v = np.array([v_x, v_y, v_p_x, v_p_y]).T
    return v

def my_potential(positions, PARAMETERS = None):
    x, y = positions.T
    # Function parameters
    # None
    # Potential energy function
    V = (1/4)*x**4 - (1/2)*x**2 + (1/2)*y**2
    return V

# Integration parameters
t0 = 0
tau = 10
# Lp-norm, p-value
p_value = 1/2
# Mesh visualisation slice parameters
H0 = 1 # Energy level
ax1_min,ax1_max = [-2, 2]
ax2_min,ax2_max = [-2, 2]
N1, N2 = [600, 600]
# Box escape condition
box_boundaries = False
# Miscellaneous grid parameters
dims_fixed = [0,1,0,0] # Variable ordering (x y p_x p_y)
dims_fixed_values = [0] # This can also be an array of values
dims_slice = [1,0,1,0] # Visualisation slice
momentum_sign = 1 # Direction of momentum that defines the slice - (1) positive / (-1) negative

potential_energy = my_potential
vector_field = my_vector_field
slice_parameters = [[ax1_min, ax1_max, N1],[ax2_min, ax2_max, N2]]
grid_parameters = {
        'slice_parameters' : slice_parameters,
        'dims_slice' : dims_slice,
        'dims_fixed' : dims_fixed,
        'dims_fixed_values' : dims_fixed_values,
        'momentum_sign' : momentum_sign,
        'potential_energy': potential_energy,
        'energy_level': H0
    }

LD_forward = compute_lagrangian_descriptor(grid_parameters, vector_field, tau, p_value)
LD_backward = compute_lagrangian_descriptor(grid_parameters, vector_field, -tau, p_value)

end = time.time()

print(end-start)

figs = draw_all_lds(LD_forward, LD_backward, grid_parameters, tau, p_value)

#draw_ld(LD_forward, 'forward', grid_parameters, tau, p_value)