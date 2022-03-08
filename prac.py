from numba import njit, cfunc, jit
from NumbaLSODA import lsoda_sig, lsoda

import numpy as np
import matplotlib.pyplot as plt
import time

a = np.array([1,2,3,4,5])

b = [1,2]