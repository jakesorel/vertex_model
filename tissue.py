from tri_functions import *
from force_functions import *
from periodic_functions import *
from growth_functions import *
import matplotlib.pyplot as plt
import time
import sys
class Tissue:
    def __init__(self):
        self.nc = None
        self.nv = None
        self.Lx, self.Ly = None, None
        self.neigh = None


    def set_global_geometry(self, Lx, Ly):
        self.Lx, self.Ly = Lx, Ly

    def update_neighbourhood(self,fast=True):
        self.neigh = get_neighbours(self.tri,neigh=self.neigh,fast=fast)
        self.v_neigh = tri_call3(self.v,self.neigh)
        self.v_CCW = roll3(self.v_neigh,1)
        self.v_CW = roll3(self.v_neigh,-1)

    def initialize_tissue(self, A0=1, noise=0.005):
        self.cnt = bounded_hexagonal_lattice(A0, self.Lx, self.Ly, noise=noise)
        self.cnt = mod2(self.cnt, self.Lx, self.Ly)
        self.nc = self.cnt.shape[0]
        self.tri, self.v = triangulate_periodic(self.cnt, self.Lx, self.Ly)
        self.tcnt = tri_call3(self.cnt,self.tri)
        self.update_neighbourhood()


    def assign_parameters(self,Kappa,Lambda,Gamma,A0,inv_viscosity,expansion_drag):
        """
        According to energy functional

        A is area, P is perimeter and l_ij is the edge length between cells i and j

        Gamma, Kappa, A0 are cell intrinsic properties i.e. vectors of n_cell
        Lambda specifies edges i.e. (symmetric) matrix of n_cell x n_cell.

        E = Sum_i Kappa_i*(A_i-A0_i)^2 + Gamma_i*P_i^2 + Sum_j Lambda_ij l_ij

        :param Kappa:
        :param Lambda:
        :param Gamma:
        :param A0:
        :return:
        """
        if Kappa is not None:
            self.Kappa = Kappa
            self.tKappa = tri_call(self.Kappa,self.tri)

        if Lambda is not None:
            self.Lambda = Lambda
            self.tLambda_CCW = tri_mat_call(self.Lambda, self.tri, direc=-1)
            self.tLambda_CW = tri_mat_call(self.Lambda, self.tri, direc=1)

        if Gamma is not None:
            self.Gamma = Gamma
            self.tGamma = tri_call(self.Gamma,self.tri)

        if A0 is not None:
            self.A0 = A0
            self.tA0 = tri_call(self.A0,self.tri)

        if inv_viscosity is not None:
            self.inv_viscosity = inv_viscosity

        if expansion_drag is not None:
            self.expansion_drag = expansion_drag


    def update_A0(self,A0):
        self.A0 = A0
        self.tA0 = tri_call(self.A0,self.tri)

    def get_displacements(self):
        self.update_neighbourhood()
        ##Calculate the displacement between vertex and its CCW neighbour (with respect to the centroid of the cell)
        ## i.e. dvn[i,j] will be the length vector along the edge that connects cell tri[i,k] to tri[i,k-1]
        self.vn = get_vn(self.v,self.v_CCW,self.Lx,self.Ly)

        ##And for the CW vertex
        self.vn_CW = get_vn(self.v,self.v_CW,self.Lx,self.Ly)


        ##Calculate the periodic displacmeent from vector from centroid to vertex
        self.vc = get_vc(self.v,self.tcnt,self.Lx,self.Ly)

        ##And for the CCW vertex
        self.vc_CCW = get_vc3(self.v_CCW,self.tcnt,self.Lx,self.Ly)
        # self.vc_CCW = get_vc3(np.expand_dims(self.v,1) - self.vn,self.tcnt,self.Lx,self.Ly)


        ##And for the CW vertex
        self.vc_CW = get_vc3(self.v_CW,self.tcnt,self.Lx,self.Ly)
        # self.vc_CW = get_vc3(np.expand_dims(self.v,1) - self.vn_CW,self.tcnt,self.Lx,self.Ly)


        ##4. Calculate edge jacobian:
        self.edge_jac = get_edge_jac(self.vc,self.vc_CCW)

        ##Calculate the distances by calculating the norm of vn
        self.tl = tnorm(self.vn)
        self.tl_CW = tnorm(self.vn_CW)

        ##Calculate the normalized direction vector of the edges.
        self.t_edgedirCCW = self.vn/np.expand_dims(self.tl,2)
        self.t_edgedirCW = self.vn_CW/np.expand_dims(self.tl_CW,2)

    def hard_update_centroids(self,hard_update_i):
        ##slow. this is for debugging only
        for i in hard_update_i:
            pts = self.v[np.where(self.tri==i)[0]]
            pts = per(pts-pts[0],self.Lx,self.Ly) + pts[0]
            self.cnt[i] = mod1(pts.mean(axis=0),self.Lx,self.Ly)
        self.tcnt = tri_call3(self.cnt,self.tri)


    def update_geometry(self):
        self.get_centroids()
        self.get_displacements()

    def get_centroids(self):
        self.get_displacements()
        if self.edge_jac.min()<0:
            concave_count = assemble_tri(1.0*(self.edge_jac<0),self.tri).ravel()
            self.hard_update_centroids(np.nonzero(concave_count>0)[0])
            self.get_displacements()
        self.cnt = assemble_tri3((self.vc+self.vc_CW)*np.expand_dims(self.edge_jac,2),self.tri)/(6*self.A) + self.cnt ##uses previous value of centroids as a reference point, important given periodicity.
        if np.isnan(self.cnt).any():
            print(self.edge_jac.min())
            print(self.A.min(),(self.A**2).min())
            print(np.nonzero(self.tri==np.nonzero(self.A.ravel()==self.A.min())[0][0]))
            print('RuntimeWarning',self.state)
            # sys.exit(1)
        self.cnt = mod2(self.cnt,self.Lx,self.Ly)
        self.tcnt = tri_call3(self.cnt,self.tri)


    def get_P(self):
        """
        Calculates edge lengths and sums them
        :return:
        """
        ##2. Assemble using neighbourhood information (tri)
        self.P = assemble_tri(self.tl,self.tri)
        self.tP = tri_call(self.P,self.tri)


    def get_A(self):
        """
        Calculates triangle areas (wrt. centroid) and sums them.

        Position of centroid is arbitrary (shoe-lace formula) provided it is sufficiently near to the vertices not to
        jeopardize cross-product calculations given periodicity
        :return:
        """

        self.A = assemble_tri(self.edge_jac/2,self.tri)##shoe-lace formula
        self.tA = tri_call(self.A,self.tri)

    def get_F(self):
        self.F = get_F(self.vc_CW,self.vc_CCW,self.tA,self.t_edgedirCW,self.t_edgedirCCW,self.tP,self.tKappa,self.tGamma,
                     self.tLambda_CCW,self.tLambda_CW,self.tA0,self.inv_viscosity)

    def get_F_tissue(self,expansion_drag=np.array((20,20))):
        self.F_tissue = get_FL(self.Lx,self.Ly,self.vn,self.t_edgedirCCW,self.tP,self.tGamma,self.tLambda_CCW,self.Kappa,self.A,self.A0,expansion_drag)


    def get_n_edges(self):
        self.n_edges = n_vertices_all(self.tri)

    def check_three_sided(self,cell_i):
        return n_vertices(cell_i,self.tri)==3

    def do_transitions(self,eps=0.1,flip_mult = 1.01):
        """
        enforce only one transition per time-step.

        :return:
        """
        short_mask = self.tl_CW<eps
        if short_mask.any():
            short_locs = np.array((np.nonzero(short_mask))).T
            tri_i, e_i = short_locs[int(np.random.random()*short_locs.shape[0])]
            chosen_tri = self.tri[tri_i]
            ##c_a and c_b flank the short edge. c_c and c_d flank the other edge involved in the T1
            c_a,c_b,c_c = np.roll(chosen_tri,-e_i)
            # print(n_vertices(c_a,self.tri),n_vertices(c_b,self.tri),n_vertices(c_c,self.tri))
            if self.check_three_sided(c_a):
                a=1
                # self.do_T2(c_a)
            elif self.check_three_sided(c_b):
                a=1
                # self.do_T2(c_b)
            else:
                self.do_T1(tri_i,e_i,c_a,c_b,c_c,eps,flip_mult)

    def do_T1(self,tri_i,e_i,c_a,c_b,c_c,eps=0.1,flip_mult = 1.01):
        """
        enforce only one T1 per time-step.

        I can draw a diagram to explain at some point.
        :param eps:
        :return:
        """

        neigh_i = self.neigh[tri_i, np.mod(e_i - 1, 3)]
        neigh_tri = self.tri[neigh_i]

        k = ((neigh_tri == c_a)*np.arange(3,dtype=np.int64)).sum()
        c_d = neigh_tri[np.mod(k+1,3)]

        ##reset the tri ids.
        self.tri[tri_i] = np.array((c_a,c_d,c_c))
        self.tri[neigh_i] = np.array((c_b,c_c,c_d))

        v_mid = self.v[tri_i] - self.vn_CW[tri_i,e_i]/2
        e_dir_new = self.t_edgedirCW[tri_i,e_i]
        e_dir_new_z = np.array((-e_dir_new[1],e_dir_new[0]))
        half_edge_length = (flip_mult*eps/2)
        v_new1= v_mid + half_edge_length*e_dir_new_z
        v_new2 = v_mid - half_edge_length*e_dir_new_z

        self.v[tri_i] = mod1(v_new1,self.Lx,self.Ly)
        self.v[neigh_i] = mod1(v_new2,self.Lx,self.Ly)

        self.neigh[self.neigh[tri_i]] = -1
        self.neigh[self.neigh[neigh_i]] = -1
        self.neigh=None
        self.update_geometry()

    def do_T2(self,cell_i):
        ###remove all tris related to this cell
        vertex_mask = get_vertex_mask(cell_i,self.tri)
        tri_i,e_i = get_vertex_trids(cell_i,self.tri)
        neighbours = self.tri[(tri_i,np.mod(e_i+1,3))]
        self.tri = self.tri[~vertex_mask]


        ##add a new vertex comprised of the three neighbours of that three-edged cell. Vertex at the centroid of the cell.
        self.tri = np.row_stack((self.tri,neighbours))
        self.tri = reset_tri_after_removal(cell_i,self.tri)
        cntroid = self.cnt[cell_i]
        self.cnt = np.row_stack((self.cnt[:cell_i],self.cnt[cell_i+1:]))
        self.tcnt = tri_call3(self.cnt,self.tri)

        self.v = np.row_stack((self.v[~vertex_mask],cntroid))

        ##re-assign geometries
        self.neigh = None
        self.get_displacements()
        self.get_A()
        self.update_geometry()

        ###NEED TO UPDATE CELL CYCLE PARAMS.

        self.nc = self.cnt.shape[0]



    def expand_tissue(self,dt):
        # F_expand = -expansion_drag*np.mean(self.F*self.v,axis=0)/np.array((self.Lx**2,self.Ly**2))
        self.get_F_tissue(self.expansion_drag)
        expand = (1+self.F_tissue*dt/np.array((self.Lx,self.Ly)))
        self.v *= expand
        self.cnt *= expand
        self.tcnt *= expand
        self.Lx,self.Ly = self.Lx*expand[0],self.Ly*expand[1]

    # def do_T2(self,alpha=):
