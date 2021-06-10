import numpy as np
from numba import jit


@jit(nopython=True)
def mod1(X,Lx,Ly):
    return np.array((np.mod(X[0],Lx),np.mod(X[1],Ly)))


@jit(nopython=True)
def mod2(X,Lx,Ly):
    return np.column_stack((np.mod(X[:,0],Lx),np.mod(X[:,1],Ly)))

@jit(nopython=True)
def mod3(X,Lx,Ly):
    return np.dstack((np.mod(X[:,:,0],Lx),np.mod(X[:,:,1],Ly)))


@jit(nopython=True)
def per1(X, Lx, Ly):
    x = np.mod(X[0] + Lx / 2, Lx) - Lx / 2
    y = np.mod(X[1] + Ly / 2, Ly) - Ly / 2
    return np.array((x, y))


@jit(nopython=True)
def per(X, Lx, Ly):
    x = np.mod(X[:, 0] + Lx / 2, Lx) - Lx / 2
    y = np.mod(X[:, 1] + Ly / 2, Ly) - Ly / 2
    return np.column_stack((x, y))

@jit(nopython=True)
def per3(X, Lx, Ly):
    x = np.mod(X[:, :, 0] + Lx / 2, Lx) - Lx / 2
    y = np.mod(X[:, :, 1] + Ly / 2, Ly) - Ly / 2
    return np.dstack((x, y))
