# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:52:27 2021

@author: sebzi
"""
import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('D:/New folder/masters/github code/ldds-develop')

import ldds
from ldds.base import compute_lagrangian_descriptor, perturb_field
from ldds.tools import draw_all_lds
from ldds.vector_fields import Duffing1D, forcing
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

alpha, beta = [1, 1]
vector_field_original = lambda t,u: Duffing1D(t, u, PARAMETERS = [alpha, beta])

# Mesh parameters to calculate LDs

# 2D Domain
x_min, x_max = [-1.6, 1.6]
y_min, y_max = [-1, 1]

# Grid size
Nx, Ny = [400, 400]

grid_parameters = [(x_min, x_max, Nx), (y_min, y_max, Ny)]

# Integration parameters (Calculate LDs at time t = t0 by integrating trajectories over the
# time interval [t0-tau,t0+tau])

# Initial time to compute LDs
t0 = 0

# Time interval half width
tau = 15

# LDs definition (we will ue the p-value seminorm)
p_value = 0.5

# Perturbation parameters
phase_shift, pert_type, amplitude, frequency = [t0, 1, 0.25, np.pi]

# Define perturbation
perturbation = lambda t,u: forcing(t, u, perturbation_params = [phase_shift, pert_type, amplitude, frequency])

# Add the perturbation to the original vector field
vector_field = perturb_field(vector_field_original , perturbation)
LD_forward = compute_lagrangian_descriptor(grid_parameters, vector_field_original, tau, p_value)
LD_backward = compute_lagrangian_descriptor(grid_parameters, vector_field_original, -tau, p_value)


figs = draw_all_lds(LD_forward, LD_backward, grid_parameters, tau, p_value)
