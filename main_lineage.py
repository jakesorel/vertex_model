from lineage import Lineage
import numpy as np
import matplotlib.pyplot as plt
from plotting_functions import *

self = Lineage()
self.set_t_span(dt=0.001,tfin=1)
self.set_cell_cycle()
self.initialize_tissue(100,A0_approx=1,Lx=8,Ly=8,inv_viscosity=50,expansion_drag=np.array((10,1)),
                       Lambda=-0.075)
self.simulate(margin=30)
plt.close("all")

plt.plot(self.t_span,(~np.isnan(self.v_save[:,:,0])).sum(axis=1))
plt.show()

(self.tissue.A.ravel() - self.tissue.A0)**2

plt.plot(self.L_save[:,0])
plt.plot(self.L_save[:,1])

plt.show()

animate_mesh(self.v_save, self.neigh_save, self.L_save, n_frames=20,scale=5)

self.animate_mesh(20)
# plt.plot(A0_save[:,0])
# plt.plot(A_save[:,0])
# plt.show()
# self.animate_mesh(20)

fig, ax = plt.subplots(1,2)
plot_mesh(ax[0], self.tissue.v, self.tissue.neigh, self.tissue.Lx, self.tissue.Ly, color="black", alpha=1)
# plot_tri(ax[0],self.tissue.tcnt,self.tissue.Lx,self.tissue.Ly,color="grey")
for i in range(self.nc):
    ax[0].scatter(self.tissue.cnt[i,0],self.tissue.cnt[i,1])

    ax[0].text(self.tissue.cnt[i,0],self.tissue.cnt[i,1],i)
fig.show()
ax[0].scatter(self.tissue.cnt[:,0],self.tissue.cnt[:,1])
self.divide(20)
plot_mesh(ax[1], self.tissue.v, self.tissue.neigh, self.tissue.Lx, self.tissue.Ly, color="black", alpha=1)
ax[1].scatter(self.tissue.cnt[:,0],self.tissue.cnt[:,1])
plot_tri(ax[1],self.tissue.tcnt,self.tissue.Lx,self.tissue.Ly,color="grey")
fig.show()

self.A0s = np.zeros((1000,self.nc))
dt = 1/1000
for i in range(1000):
    self.tc += dt
    self.tc = np.mod(self.tc,1)
    self.update_cell_cycle_states()
    self.update_A0()
    self.A0s[i] = self.A0

plt.plot(self.A0s)
plt.show()