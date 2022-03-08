# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 11:43:56 2022

@author: sebzi
"""

import numba
import numpy as np

a = np.array([1.2,2,3,4])
b = np.array([ [1,2.2],[3.3,4]])
print(numba.float64[::1])

u0 = np.array([1.1,2,3,4])

print(numba.typeof(u0))