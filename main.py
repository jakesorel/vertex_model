from tissue import *
from plotting_functions import *
self = Tissue()
self.set_global_geometry(6,7)
self.initialize_tissue(1.25,1e-1)
self.assign_parameters(Kappa = np.ones(self.nc),
                       Gamma = np.ones(self.nc)*0.04,#0.04
                       Lambda = np.ones((self.nc,self.nc))*0.075,#0.075
                       A0 = np.ones((self.nc))*1.25)

dt = 0.05
tfin = 100
t_span = np.arange(0,tfin,dt)
nt = t_span.size


v_save = np.zeros((nt,self.tri.shape[0],2))
tri_save = np.zeros((nt,self.tri.shape[0],3),dtype=np.int64)
neigh_save = np.zeros((nt,self.tri.shape[0],3),dtype=np.int64)

self.get_displacements()
self.get_A()
self.get_P()

# nt = 100
t0 = time.time()
for i in range(nt):
    self.do_T1(eps=0.1)
    self.get_centroids()
    self.get_displacements()
    self.get_P()
    self.get_A()
    self.get_F()
    self.v += self.F*dt
    self.v = mod2(self.v,self.Lx,self.Ly)
    v_save[i] = self.v
    tri_save[i] = self.tri
    neigh_save[i] = self.neigh
t1=time.time()
print((t1-t0),"s",(t1-t0)/nt * 10**6, "ns per step")


animate_mesh(v_save,neigh_save,self.Lx,self.Ly,n_frames=20)


fig, ax = plt.subplots(1,3)
plot_mesh(ax[0],self.v,self.neigh,self.Lx,self.Ly)
plot_mesh(ax[2],self.v,self.neigh,self.Lx,self.Ly,color="red",alpha=0.5)

self.do_T1(eps=0.2)
self.get_displacements()
plot_mesh(ax[1],self.v,self.neigh,self.Lx,self.Ly)

plot_mesh(ax[2],self.v,self.neigh,self.Lx,self.Ly,alpha=0.5)
ax[1].scatter(self.v[:,0],self.v[:,1])
# ax[1].scatter(self.v[:,0],self.v[:,1])
fig.show()

#
# i = 30
# plt.scatter(self.cnts[:,0],self.cnts[:,1],color="grey",alpha=0.2)
# plt.scatter(self.v[:,0],self.v[:,1],color="darkgrey",alpha=0.2)
#
# plt.scatter(self.v[i,0],self.v[i,1])
# xt = self.cnts[self.tri]
# xt = roll3(xt,1)
# # xt = np.dstack((roll_reverse(xt[:,:,0]),roll_reverse(xt[:,:,1])))
# for j in range(3):
#     plt.scatter(self.v_neigh[i,j,0],self.v_neigh[i,j,1])
#     plt.scatter(xt[i,j,0],xt[i,j,1],color="purple")
#     plt.text(self.v_neigh[i,j,0],self.v_neigh[i,j,1],j,fontsize=20)
#
#     plt.text(xt[i,j,0],xt[i,j,1],j,fontsize=20,color="purple")
#
# plt.show()