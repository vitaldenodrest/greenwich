'''
Example script for plotting a 1D -> 2D function.
Our example is a complex-valued of a real variable
'''

import matplotlib.pyplot as plt
import cmath
import numpy as np

# Example function
k=10
def u(x):
    return cmath.exp(k*x*1j)

# Evaluation points
X = np.linspace(start=0, stop=1, num=100)

# Values
f_X = np.vectorize(u)(X)
f_X_real = np.real(f_X)
f_X_imag = np.imag(f_X)
f_X_arg = np.angle(f_X)
f_X_abs = np.absolute(f_X)

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(X, f_X_real, f_X_imag)
plt.show()

# Real plot
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(X, f_X_real)
plt.show()

# Imaginary plot
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(X, f_X_imag)
plt.show()

# Arg plot
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(X, f_X_arg)
plt.show()

# Abs plot
