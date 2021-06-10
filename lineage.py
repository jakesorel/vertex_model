import numpy as np
from tissue import Tissue
from growth_functions import *
import time
from periodic_functions import *
from plotting_functions import *

"""
Alternative strategy. 
Fill lineage with all potential cells, live, dead or yet to be born. 
Then update the actual params of the "tissue" class with only the live cells. 
Feed information back to the lineage class r.e. lineage dynamics (birth, death) 

So move the divide function to the tissue class. 
 
Would be good to have some generic function that can record all of the cell data over time. 



"""


class Lineage:
    def __init__(self):
        self.dt,self.tfin,self.t_span = None,None,None
        self.nc = None

    def set_t_span(self,dt=0.05,tfin=100):
        self.dt = dt
        self.tfin = tfin
        self.t_span = np.arange(0, tfin, dt)
        self.nt = self.t_span.size


    def set_cell_cycle(self,t_G1=0.4,t_S=0.333,t_G2=0.166,t_cc=1,Ac=1.3):
        self.Ac = Ac
        self.t_G1 = t_G1  # Time proportion in G1 phase
        self.t_S = t_S  # Time proportion in S phase
        self.t_G2 = t_G2  # Time proportion in G2 phase
        self.t_M = t_cc - (self.t_G1 + self.t_S + self.t_G2)
        self.t_I = t_G1 + t_S + t_G2 #Time proprotion in interphase
        assert self.t_M>0,"t_G1 + t_S + t_G2 must be less than 1 (time units are normalized to cell-cycle time"

    def initialize_cell_cycles(self,g_av=1,g_sig=0.2):
        """
        tc is the age of the cell
        :return:
        """
        # self.tc0 = np.random.uniform(0,1,self.nc)
        self.g_av = g_av
        self.g_sig = g_sig
        self.tc = np.random.uniform(0,1,self.nc)
        self.g = np.random.normal(self.g_av,self.g_sig,self.nc)

    def update_cell_cycle_states(self):
        self.cc_state = get_cell_cycle_state(self.tc, self.t_G1, self.t_S, self.t_G2)

    def update_A0(self):
        self.update_cell_cycle_states()
        self.A0 = get_A0(self.tc,self.g,self.cc_state,self.t_G1,self.t_S,self.t_G2)
        self.tissue.update_A0(self.A0)

    def ready_to_divide(self):
        self.divide_mask = ready_to_divide(self.tc, self.tissue.A.ravel(), self.Ac, self.t_I)
        return self.divide_mask

    def perform_division(self,direc="random"):
        self.ready_to_divide()
        while self.divide_mask.any():
            dividing_cells = np.nonzero(self.divide_mask)[0]
            dividing_cell = dividing_cells[int(np.random.random()*dividing_cells.size)]
            self.divide(dividing_cell,direc=direc)
            self.update_A0()
            self.ready_to_divide()

    def initialize_tissue(self, nc,A0_approx=2,Gamma=0.04,Lambda=0.075,Kappa=1,Lx=4,Ly=4,inv_viscosity=50,expansion_drag=np.array((50,1)),
                          g_av=1,g_sig=0.2):
        self.mechanical_params = {"Kappa":Kappa,"Gamma":Gamma,"Lambda":Lambda}
        self.nc = nc
        self.tissue = Tissue()
        self.tissue.set_global_geometry(Lx,Ly)
        self.tissue.initialize_tissue(A0_approx, 1e-1)
        self.nc = self.tissue.nc
        self.initialize_cell_cycles(g_av=g_av,g_sig=g_sig)
        self.update_A0()
        self.assign_mechanical_params(inv_viscosity=inv_viscosity,expansion_drag=expansion_drag)


    def assign_mechanical_params(self,inv_viscosity=None,expansion_drag=None):
        self.tissue.assign_parameters(Kappa=np.ones(self.nc)*self.mechanical_params["Kappa"],
                               Gamma=np.ones(self.nc) * self.mechanical_params["Gamma"],  # 0.04
                               Lambda=np.ones((self.nc, self.nc)) * self.mechanical_params["Lambda"],  # 0.075
                               A0=None,inv_viscosity=inv_viscosity,expansion_drag=expansion_drag)

    def simulate(self,margin=2):
        self.v_save = np.ones((self.nt, self.tissue.tri.shape[0]*margin, 2))*np.nan
        self.tri_save = np.ones((self.nt, self.tissue.tri.shape[0]*margin, 3), dtype=np.int64)*-1
        self.neigh_save = np.ones((self.nt, self.tissue.tri.shape[0]*margin, 3), dtype=np.int64)*-1
        self.L_save = np.ones((self.nt,2))
        self.tissue.get_displacements()
        self.tissue.get_A()

        # nt = 100
        t0 = time.time()
        for i in range(self.nt):
            self.tc += self.dt
            # self.tc = np.mod(self.tc,1)
            # self.tc = np.zeros_like(self.tc)
            # self.g = np.ones_like(self.g)
            self.update_A0()

            self.tissue.state = 1
            self.tissue.update_geometry()
            self.tissue.do_transitions(eps=0.1)
            self.tissue.state = 2
            self.perform_division()


            self.tissue.get_P()
            self.tissue.get_A()
            self.tissue.get_F()
            self.tissue.v += self.tissue.F * self.dt
            self.tissue.v = mod2(self.tissue.v, self.tissue.Lx, self.tissue.Ly)
            self.tissue.expand_tissue(self.dt)
            self.v_save[i,:self.tissue.v.shape[0]] = self.tissue.v
            self.tri_save[i,:self.tissue.tri.shape[0]] = self.tissue.tri
            self.neigh_save[i,:self.tissue.neigh.shape[0]] = self.tissue.neigh
            self.L_save[i] = self.tissue.Lx,self.tissue.Ly
        t1 = time.time()
        print((t1 - t0), "s", (t1 - t0) / self.nt * 10 ** 6, "ns per step")
        return

    def divide(self,cell_i,direc="random"):
        self.tissue.get_displacements()
        self.tissue.v,self.tissue.tri,self.tissue.cnt = divide_cell(cell_i,self.tissue.tri,
                                  self.tissue.v,self.tissue.vc,self.tissue.vc_CCW,
                                  self.tissue.cnt, self.tissue.Lx,self.tissue.Ly,
                                  direc=direc)
        self.tissue.tcnt = tri_call3(self.tissue.cnt,self.tissue.tri)
        self.tissue.neigh = None
        # self.tissue.update_geometry()
        self.tissue.update_neighbourhood(fast=False)
        self.tissue.get_displacements() ##needs to be called with approximate centroids
        self.tissue.get_A()
        self.tissue.update_geometry()
        # self.tissue.get_centroids()
        # self.tissue.get_displacements() ##and adjusted for proper centroids

        ##reset clocks
        self.tc[cell_i] = 0
        self.tc = np.append(self.tc,0)

        ##reset growth rates
        self.g[cell_i] = np.random.normal(self.g_av,self.g_sig) ##re-generate growth rate of the new cell in place of cell_i
        self.g = np.append(self.g,np.random.normal(self.g_av,self.g_sig)) ##and for the new cell at position nc+1

        self.nc +=1

        self.assign_mechanical_params()


    def animate_mesh(self,n_frames):
        animate_mesh(self.v_save, self.neigh_save, self.L_save, n_frames=n_frames)


"""
To do: 

-- Implement division (and lineage tracking) 
-- Implement domain expansion
-- Implement the FEM
-- Implement the PONI network 
-- Implement 
"""

