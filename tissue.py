from tri_functions import *
from force_functions import *
from periodic_functions import *
import matplotlib.pyplot as plt
import time
class Tissue:
    def __init__(self):
        self.nc = None
        self.nv = None
        self.Lx, self.Ly = None, None
        self.neigh = None


    def set_global_geometry(self, Lx, Ly):
        self.Lx, self.Ly = Lx, Ly

    def update_neighbourhood(self):
        self.neigh = get_neighbours(self.tri,neigh=None)
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


    def assign_parameters(self,Kappa,Lambda,Gamma,A0):
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
        self.Kappa,self.Lambda,self.Gamma,self.A0 = Kappa,Lambda,Gamma,A0

        self.tKappa,self.tGamma,self.tA0 = tri_call(self.Kappa,self.tri),tri_call(self.Gamma,self.tri),tri_call(self.A0,self.tri)

        ##tLambda is the value of Lambda_ij for the edge that connects a vertex to its CCW neighbour
        self.tLambda_CCW = tri_mat_call(self.Lambda,self.tri,direc=-1)
        self.tLambda_CW = tri_mat_call(self.Lambda,self.tri,direc=1)

    def get_displacements(self):
        self.update_neighbourhood()
        ##Calculate the displacement between vertex and its CCW neighbour (with respect to the centroid of the cell)
        ## i.e. dvn[i,j] will be the length vector along the edge that connects cell tri[i,k] to tri[i,k-1]
        self.vn = get_vn(self.v,self.v_CCW,self.Lx,self.Ly)

        ##And for the CW vertex
        self.vn_CW = get_vn(self.v,self.v_CW,self.Lx,self.Ly)


        ##Calculate the distances by calculating the norm of vn
        self.tl = tnorm(self.vn)
        self.tl_CW = tnorm(self.vn_CW)


        ##Calculate the normalized direction vector of the edges.
        self.t_edgedirCCW = self.vn/np.expand_dims(self.tl,2)
        self.t_edgedirCW = self.vn_CW/np.expand_dims(self.tl_CW,2)

        ##Calculate the periodic displacmeent from vector from centroid to vertex
        self.vc = get_vc(self.v,self.tcnt,self.Lx,self.Ly)

        ##And for the CCW vertex
        # self.vc_CCW = get_vc3(self.v_CCW,self.tcnt,self.Lx,self.Ly)
        self.vc_CCW = get_vc3(np.expand_dims(self.v,1) - self.vn,self.tcnt,self.Lx,self.Ly)


        ##And for the CW vertex
        # self.vc_CW = get_vc3(self.v_CW,self.tcnt,self.Lx,self.Ly)
        self.vc_CW = get_vc3(np.expand_dims(self.v,1) - self.vn_CW,self.tcnt,self.Lx,self.Ly)


        ##4. Calculate edge jacobian:
        self.edge_jac = get_edge_jac(self.vc,self.vc_CCW)

    def update_centroids(self):
        ##slow. this is for debugging only
        for i in range(self.cnt.shape[0]):
            pts = self.v[np.where(self.tri==i)[0]]
            pts = per(pts-pts[0],self.Lx,self.Ly) + pts[0]
            self.cnt[i] = mod1(pts.mean(axis=0),self.Lx,self.Ly)
        self.tcnt = tri_call3(self.cnt,self.tri)



    def get_centroids(self):
        self.cnt = assemble_tri3((self.vc+self.vc_CW)*np.expand_dims(self.edge_jac,2),self.tri)/(6*self.A) + self.cnt ##uses previous value of centroids as a reference point, important given periodicity.
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
                     self.tLambda_CCW,self.tLambda_CW,self.tA0)
        # tdA_dv = (self.vc_CW-self.vc_CCW)/2
        # tdA_dv = np.dstack((-tdA_dv[:,:,1],+tdA_dv[:,:,0]))
        # tdEA_dA = self.tKappa*(self.tA-self.tA0)
        # tdEA_dv = np.expand_dims(tdEA_dA,2)*tdA_dv
        #
        # tdP_dv = (self.t_edgedirCW + self.t_edgedirCCW)
        # tdEP_dP = self.tGamma*self.tP
        # tdEP_dv = np.expand_dims(tdEP_dP,2)*tdP_dv
        #
        # tdEl_dv = np.expand_dims(self.tLambda_CCW,2)*self.t_edgedirCCW + np.expand_dims(self.tLambda_CW,2)*self.t_edgedirCW
        #
        # tdE_dv = tdEA_dv + tdEP_dv + tdEl_dv
        # self.tF = -tdE_dv
        # self.F = self.tF[:,0] + self.tF[:,1] + self.tF[:,2]
        # dE_dv = assemble_tri3(tdE_dv,self.tri)

    def do_T1(self,eps=0.1):
        """
        enforce only one T1 per time-step.

        I can draw a diagram to explain at some point.
        :param eps:
        :return:
        """
        short_mask = self.tl_CW<eps
        if short_mask.any():
            short_locs = np.array((np.nonzero(short_mask))).T
            tri_i, e_i = short_locs[int(np.random.random()*short_locs.shape[0])]
            chosen_tri = self.tri[tri_i]
            neigh_i = self.neigh[tri_i,np.mod(e_i-1,3)]
            neigh_tri = self.tri[neigh_i]
            # print(neigh_tri)
            # set(list(neigh_tri)).difference(set(list(chosen_tri)))

            ##c_a and c_b flank the short edge. c_c and c_d flank the other edge involved in the T1
            c_a,c_b,c_c = np.roll(chosen_tri,-e_i)
            k = ((neigh_tri == c_a)*np.arange(3,dtype=np.int64)).sum()
            c_d = neigh_tri[np.mod(k+1,3)]

            ##reset the tri ids.
            self.tri[tri_i] = np.array((c_a,c_d,c_c))
            self.tri[neigh_i] = np.array((c_b,c_c,c_d))

            v_mid = self.v[tri_i] - self.vn_CW[tri_i,e_i]/2
            e_dir_new = self.t_edgedirCW[tri_i,e_i]
            e_dir_new_z = np.array((-e_dir_new[1],e_dir_new[0]))
            v_new1= v_mid + (1.01*eps/2)*e_dir_new_z
            v_new2 = v_mid - (1.01*eps/2)*e_dir_new_z
            # print(np.dot(v_new2-v_new1,self.v[tri_i]-self.v[neigh_i]))

            self.v[tri_i] = mod1(v_new1,self.Lx,self.Ly)
            self.v[neigh_i] = mod1(v_new2,self.Lx,self.Ly)



            #
            self.neigh[self.neigh[tri_i]] = -1
            self.neigh[self.neigh[neigh_i]] = -1
            self.neigh=None
            self.update_neighbourhood()
            print("T1")

            # fig, ax = plt.subplots()
            # ax.scatter(self.v[tri_i,0],self.v[tri_i,1],color="k")
            # ax.scatter(self.v[neigh_i,0],self.v[neigh_i,1],color="blue")
            #
            # ax.plot((self.v[tri_i, 0],self.v[neigh_i, 0]), (self.v[tri_i, 1],self.v[neigh_i, 1]), color="k")
            # # ax.scatter(self.v[neigh_i, 0], self.v[neigh_i, 1], color="blue")
            #
            # ax.scatter(v_mid[0],v_mid[1],color="purple")
            #
            # ax.scatter(v_new1[0],v_new1[1],color="red")
            # ax.scatter(v_new2[0],v_new2[1],color="green")
            # ax.plot(np.array((v_new1,v_new2))[:,0],np.array((v_new1,v_new2))[:,1], color="k")
            #
            # fig.show()
            # ##re-set the neighbourhood ids.
            # new_chosen_neigh = np.array((neigh_i,self.neigh[tri_i,np.mod(e_i+1,3)],self.neigh[neigh_i,np.mod(k-1,3)]))
            # new_neigh_neigh = np.array((tri_i,self.neigh[neigh_i,k],self.neigh[tri_i,e_i]))
            # self.neigh[tri_i] = new_chosen_neigh
            # self.neigh[neigh_i] = new_neigh_neigh
            # self.update_neighbourhood(reset_neigh=False)

