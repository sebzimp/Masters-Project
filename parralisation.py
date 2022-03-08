from numba import njit, cfunc, jit
from NumbaLSODA import lsoda_sig, lsoda

import numpy as np
import matplotlib.pyplot as plt
import time

start = time.perf_counter()


a1 = 0
b1 = 0.495
M1 = 2.05 * 10 ** 10
a2 = 7.258
b2 = 0.520
M2 = 25.47 * 10 ** 10
qa = 1.2
qb = 0.9
Ohm = 60
G = 4.3009 * 10 ** (-6)


# equation of motion
@cfunc(lsoda_sig)
def Hz0(t, a, da, p):  # a,t

    da[0] = a[2] + Ohm * a[1]

    da[1] = a[3] - Ohm * a[0]

    da[2] = Ohm * a[3] - a[0] * G * (M1 * (a[0] ** 2 + a[1] ** 2 / qa ** 2 + (a1 + b1) ** 2) ** (-1.5) + M2 * (
                a[0] ** 2 + a[1] ** 2 / qa ** 2 + (a2 + b2) ** 2) ** (-1.5))

    da[3] = -Ohm * a[2] - (a[1] / qa ** 2) * G * (
                M1 * (a[0] ** 2 + a[1] ** 2 / qa ** 2 + (a1 + b1) ** 2) ** (-1.5) + M2 * (
                    a[0] ** 2 + a[1] ** 2 / qa ** 2 + (a2 + b2) ** 2) ** (-1.5))


# EOM in backwards time
@cfunc(lsoda_sig)
def Hz1(t, a, da, p):  # a,t

    da[0] = -(a[2] + Ohm * a[1])

    da[1] = -(a[3] - Ohm * a[0])

    da[2] = -(Ohm * a[3] - a[0] * G * (M1 * (a[0] ** 2 + a[1] ** 2 / qa ** 2 + (a1 + b1) ** 2) ** (-1.5) + M2 * (
                a[0] ** 2 + a[1] ** 2 / qa ** 2 + (a2 + b2) ** 2) ** (-1.5)))

    da[3] = -(-Ohm * a[2] - (a[1] / qa ** 2) * G * (
                M1 * (a[0] ** 2 + a[1] ** 2 / qa ** 2 + (a1 + b1) ** 2) ** (-1.5) + M2 * (
                    a[0] ** 2 + a[1] ** 2 / qa ** 2 + (a2 + b2) ** 2) ** (-1.5)))


#@jit
def vect(x, y, px, py):
    v1 = px + Ohm * y
    v2 = py - Ohm * x
    v3 = Ohm * py - x * G * (M1 * (x ** 2 + y ** 2 / qa ** 2 + (a1 + b1) ** 2) ** (-1.5) + M2 * (
                x ** 2 + y ** 2 / qa ** 2 + (a2 + b2) ** 2) ** (-1.5))
    v4 = -Ohm * px - (y / qa ** 2) * G * (M1 * (x ** 2 + y ** 2 / qa ** 2 + (a1 + b1) ** 2) ** (-1.5) + M2 * (
                x ** 2 + y ** 2 / qa ** 2 + (a2 + b2) ** 2) ** (-1.5))

    return [v1, v2, v3, v4]


# potential
#@jit
def potential(x, y, z, PARAMETERS=[a1, b1, M1, a2, b2, M2, qa, Ohm, G]):
    V = -G * M1 * (x ** 2 + y ** 2 / qa ** 2 + (a1 + (z ** 2 / qb ** 2 + b1 ** 2) ** (0.5)) ** 2) ** (-0.5) - G * M2 * (
                x ** 2 + y ** 2 / qa ** 2 + (a2 + (z ** 2 / qb ** 2 + b2 ** 2) ** (0.5)) ** 2) ** (-0.5)
    return V


# Hamiltonian
#@jit
def Hamiltonian(x, y, px, py):
    H = 0.5 * (px ** 2 + py ** 2) + potential(x, y, 0) - Ohm * (x * py - y * px)
    return H


#@jit
def vect2(x, y, px, py):
    v1 = -(px + Ohm * y)
    v2 = -(py - Ohm * x)
    v3 = -(Ohm * py - x * G * (M1 * (x ** 2 + y ** 2 / qa ** 2 + (a1 + b1) ** 2) ** (-1.5) + M2 * (
                x ** 2 + y ** 2 / qa ** 2 + (a2 + b2) ** 2) ** (-1.5)))
    v4 = -(-Ohm * px - (y / qa ** 2) * G * (M1 * (x ** 2 + y ** 2 / qa ** 2 + (a1 + b1) ** 2) ** (-1.5) + M2 * (
                x ** 2 + y ** 2 / qa ** 2 + (a2 + b2) ** 2) ** (-1.5)))

    return [v1, v2, v3, v4]


# Energy
A = 43950
H0 = -4.18 * A  # hamiltonian

# fixed coordinate value

x0 = 0

res = 800
# grid on which LDs are calculated
ax1_min, ax1_max = [2.75, 4]
ax2_min, ax2_max = [-90, 90]
N1, N2 = [res, res]

grid_parameters = [[ax1_min, ax1_max, N1], [ax2_min, ax2_max, N2]]

x_min, x_max, Nx = grid_parameters[0]
y_min, y_max, Ny = grid_parameters[1]
points_x = np.linspace(x_min, x_max, Nx)
points_y = np.linspace(y_min, y_max, Ny)


# empty arrays for plots


# integration time
T = 7000  # timesteps
t = np.linspace(0.0, 7.0, T)

# calculating the LDs

def xrangeLD(xrange): #want xrange an array which we divided into the number of processors from the original domain
    LD = []
    LD2 = []
    for i in range(len(xrange)):
        for j in range(len(points_y)):


            y0 = xrange[i]  # x coordinate initial position
            py0 = points_y[j]

            delta = Ohm ** 2 * y0 ** 2 - 2 * (0.5 * py0 ** 2 + potential(x0, y0, 0) - H0)

            if delta >= 0:

                px0 = -Ohm * y0 + np.sqrt(delta)


                funcptr = Hz0.address  # address to ODE function for

                u0 = np.array([x0, y0, px0, py0])  # Initial conditions

                usol, success = lsoda(funcptr, u0, t, rtol=1.0e-8, atol=1.0e-9)  # solving EoM

                x = []
                y = []
                px = []
                py = []

                for k in range(len(usol)):
                    x.append(usol[k][0])
                    y.append(usol[k][1])
                    px.append(usol[k][2])
                    py.append(usol[k][3])

                #               break

                v = vect(np.array(x), np.array(y), np.array(px), np.array(py))

                # cacluating the LD with p-norm, p =0.5
                intermedLD = np.sum(0.001 * np.abs(v) ** 0.5, axis=1)
                LD.append(np.sum(intermedLD))

                funcptr2 = Hz1.address  # back

                usol2, success = lsoda(funcptr2, u0, t, rtol=1.0e-8, atol=1.0e-9)  # solving EoM

                x2 = []
                y2 = []
                px2 = []
                py2 = []


                for k in range(len(usol2)):
                    x2.append(usol2[k][0])
                    y2.append(usol2[k][1])
                    px2.append(usol2[k][2])
                    py2.append(usol2[k][3])



                v2 = vect2(np.array(x2), np.array(y2), np.array(px2), np.array(py2))

                intermedLD2 = np.sum(0.001 * np.abs(v2) ** 0.5, axis=1)
                LD2.append(np.sum(intermedLD2))
    LDprop = np.add(LD, LD2)

    return [LD, LD2, LDprop]



import multiprocessing as mp

def main():
    pool = mp.Pool(mp.cpu_count())
    x_split=np.array_split(points_x,mp.cpu_count())

    result = pool.map(xrangeLD, x_split)

    print(len(result))
    print(len(result[0]))
    print(len(result[0][0]))

    forLD = np.array([])
    backLD = np.array([])
    propLD = np.array([])
    for i in range(len(result)):
        forLD = np.concatenate((forLD,np.array(result[i][0])))
        backLD = np.concatenate((backLD,np.array(result[i][1])))
        propLD = np.concatenate((propLD,np.array(result[i][2])))

    xax = []
    yax = []

    for i in range(len(points_x)):
        for j in range(len(points_y)):

            y0 = points_x[i]  # x coordinate initial position
            py0 = points_y[j]

            delta = Ohm ** 2 * y0 ** 2 - 2 * (0.5 * py0 ** 2 + potential(x0, y0, 0) - H0)

            if delta >= 0:

                xax.append(y0)
                yax.append(py0)

    np.savetxt("grid800tau7E4.18LD.txt",propLD)
    np.savetxt("grid800tau7E4.18forLD.txt",forLD)
    np.savetxt("grid800tau7E4.18backLD.txt",backLD)

    yax = np.true_divide(yax,209.64 )
    plt.figure(dpi = 200)
    plt.scatter(xax,yax,c=propLD ,cmap = "plasma", s = 0.5)
    plt.colorbar(label = "LD")

    plt.figure(dpi = 200)
    plt.scatter(xax,yax,c=forLD ,cmap = "plasma", s = 0.5)
    plt.colorbar(label = "Forward LD")

    plt.figure(dpi = 200)
    plt.scatter(xax,yax,c=backLD ,cmap = "plasma", s = 0.5)
    plt.colorbar(label = "Backward LD")


if __name__ == "__main__":
  main()
  end = time.perf_counter()
  print(end - start)
  plt.show()



