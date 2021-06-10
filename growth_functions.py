import numpy as np
from numba import jit
from tri_functions import *
from periodic_functions import *

def get_cell_cycle_state(tc, t_G1,t_S,t_G2):
    """
    states:
    0 = G1
    1 = S
    2 = G2
    3 = M

    :param tc:
    :param t_G1:
    :param t_S:
    :param t_G2:
    :return:
    """
    state = np.zeros_like(tc,dtype=np.int64)
    state += tc > t_G1
    state += tc > (t_S + t_G1)
    state += tc > (t_G2 + t_G1 + t_S)
    return state

@jit(nopython=True)
def get_A0(tc,g,cc_state,t_G1,t_S,t_G2):
    rho = (cc_state==0)*(1-tc/t_G1) + (cc_state==2)*((tc-t_G1-t_S)/t_G2) + (cc_state==3)
    A0 = 0.5*(tc*g + 1)*(1+rho**2)
    return A0

@jit(nopython=True)
def ready_to_divide(tc,A,Ac,t_I):
    return (tc > t_I)*(A>Ac)

@jit(nopython=True)
def line_intersect(direc,vc,vc_CCW):
    """
    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    :param direc:
    :param vc:
    :param vc_CCW:
    :return:
    """
    dir_vec = np.array((np.cos(direc),np.sin(direc)))

    num = np.column_stack((-vc,vc-vc_CCW))
    denom = np.column_stack((-dir_vec,vc-vc_CCW))

    t = np.linalg.det(num)/np.linalg.det(denom)
    intersect = t*dir_vec
    return intersect

def divide_cell(cell_i,tri,v,vc,vc_CCW, cnt, Lx,Ly,direc="random",eps=0.01):
    """
    Calculates the new coordinates of the vertices, and the edges that this breaks up.

    Achieves via assigning a "direction", which may be "random" (where a random angle is chosen between 0 and 2pi)


    :param cell_i:
    :param tri:
    :param vc:
    :param vc_CCW:
    :param cnt:
    :param direc:
    :return:
    """
    # cell_i, tri, vc, vc_CCW, cnt, Lx, Ly, direc = cell_i,self.tissue.tri,self.tissue.vc,self.tissue.vc_CCW,self.tissue.cnt, self.tissue.Lx,self.tissue.Ly,"random"
    new_i = tri.max() + 1
    tri_i,e_i = get_vertex_trids(cell_i,tri)
    flat_loc = tri_i*3 + e_i
    vcs = np.column_stack((vc[:,:,0].take(flat_loc),vc[:,:,1].take(flat_loc)))
    vc_CCWs = np.column_stack((vc_CCW[:,:,0].take(flat_loc),vc_CCW[:,:,1].take(flat_loc)))


    if direc == "random":
        direc = np.random.uniform(0,np.pi*2)

    v_angles = np.mod(np.arctan2(vcs[:,1],vcs[:,0])-direc +np.pi,np.pi*2)-np.pi
    vn_angles = np.mod(np.arctan2(vc_CCWs[:,1],vc_CCWs[:,0])-direc + np.pi,np.pi*2)-np.pi

    major_edge_mask = (v_angles < 0)*(vn_angles > 0)
    minor_edge_mask = (v_angles > 0)*(vn_angles < 0)

    maj_vc,maj_vcCCW = vcs[major_edge_mask].ravel(),vc_CCWs[major_edge_mask].ravel()
    min_vc,min_vcCCW = vcs[minor_edge_mask].ravel(),vc_CCWs[minor_edge_mask].ravel()

    maj_intersect = line_intersect(direc,maj_vc,maj_vcCCW) + cnt[cell_i]
    min_intersect = line_intersect(direc,min_vc,min_vcCCW) + cnt[cell_i]
    maj_intersect = mod1(maj_intersect,Lx,Ly)
    min_intersect = mod1(min_intersect,Lx,Ly)
    major,minor = ((tri_i[major_edge_mask][0],e_i[major_edge_mask][0]),maj_intersect),((tri_i[minor_edge_mask][0],e_i[minor_edge_mask][0]),min_intersect)

    neigh_maj = tri[major[0][0],np.mod(major[0][1]-1,3)]
    neigh_min = tri[minor[0][0],np.mod(minor[0][1]-1,3)]

    maj_tri = np.array((new_i,neigh_maj,cell_i))
    min_tri = np.array((cell_i,neigh_min,new_i))
    v = np.row_stack((v,maj_intersect))
    v = np.row_stack((v,min_intersect))

    CCW_vertices = v_angles < 0
    tri[(tri_i[CCW_vertices],e_i[CCW_vertices])] = new_i
    tri = np.row_stack((tri,maj_tri))
    tri = np.row_stack((tri,min_tri))

    direc_z = np.array((-np.sin(direc),np.cos(direc)))

    cnt = np.row_stack((cnt,cnt[cell_i]))
    cnt[cell_i] -= direc_z*eps
    cnt[new_i] += direc_z*eps

    return v,tri,cnt

@jit(nopython=True)
def reset_val_after_removal(cell_i,val):
    return np.concatenate((val[:cell_i],val[cell_i+1:]))

@jit(nopython=True)
def reset_tri_after_removal(cell_i,tri):
    tri_flat = tri.ravel()
    tri_flat = tri_flat * (tri_flat<cell_i) + (tri_flat-1)*(tri_flat>cell_i)
    return tri_flat.reshape(tri.shape)
    # #
    # ##debugging plotting code for ^^
    # #
    # c = self.tissue.cnt[cell_i]
    # fig, ax = plt.subplots()
    # plot_mesh(ax, self.tissue.v, self.tissue.neigh, self.tissue.Lx, self.tissue.Ly, color="black", alpha=1)
    #
    # # major_edge = np.array((vcs[major_edge_mask].ravel()+c,vc_CCWs[major_edge_mask].ravel()+c))
    # minor_edge = np.array((vcs[minor_edge_mask].ravel()+c,vc_CCWs[minor_edge_mask].ravel()+c))
    #
    # # ax.plot(major_edge[:,0],major_edge[:,1])
    # ax.plot(minor_edge[:,0],minor_edge[:,1])
    # line = np.linspace(-1,1)
    #
    # ax.plot(line*np.cos(direc)+c[0],line*np.sin(direc)+c[1])
    # ax.scatter(maj_intersect[0],maj_intersect[1])
    # ax.scatter(min_intersect[0],min_intersect[1])
    #
    # for i in tri_i:
    #     for j in range(3):
    #         c = self.tissue.cnt[self.tissue.tri[i,j]]
    #         ax.text(c[0],c[1],self.tissue.tri[i,j])
    #
    # vs = self.tissue.v[tri_i]
    # for v in vs[v_angles<0]:
    #     ax.scatter(v[0],v[1])
    #
    #
    # fig.show()
