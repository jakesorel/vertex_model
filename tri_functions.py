import numpy as np
from numba import jit
import triangle as tr
from periodic_functions import per3,per,mod2,mod3
from scipy.sparse import coo_matrix

def order_tris(tri):
    """
    For each triangle (i.e. row in **tri**), order cell ids in ascending order
    :param tri: Triangulation (n_v x 3) np.int32 array
    :return: the ordered triangulation
    """
    nv = tri.shape[0]
    for i in range(nv):
        Min = np.argmin(tri[i])
        tri[i] = tri[i,Min],tri[i,np.mod(Min+1,3)],tri[i,np.mod(Min+2,3)]
    return tri


def remove_repeats(tri, n_c):
    """
    For a given triangulation (nv x 3), remove repeated entries (i.e. rows)

    The triangulation is first re-ordered, such that the first cell id referenced is the smallest. Achieved via
    the function order_tris. (This preserves the internal order -- i.e. CCW)

    Then remove repeated rows via lexsort.

    NB: order of vertices changes via the conventions of lexsort

    Inspired by...
    https://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array

    :param tri: (nv x 3) matrix, the triangulation
    :return: triangulation minus the repeated entries (nv* x 3) (where nv* is the new # vertices).
    """
    tri = order_tris(np.mod(tri, n_c))
    sorted_tri = tri[np.lexsort(tri.T), :]
    row_mask = np.append([True], np.any(np.diff(sorted_tri, axis=0), 1))
    return sorted_tri[row_mask]


def make_y(x,Lgrid_xy):
    """
    Makes the (9) tiled set of coordinates used to perform the periodic triangulation.

    :param x: Cell centroids (n_c x 2) np.float32 array
    :param Lgrid_xy: (9 x 2) array defining the displacement vectors for each of the 9 images of the tiling
    :return: Tiled set of coordinates (9n_c x 2) np.float32 array
    """
    n_c = x.shape[0]
    y = np.empty((n_c*9,x.shape[1]))
    for k in range(9):
        y[k*n_c:(k+1)*n_c] = x + Lgrid_xy[k]
    return y



@jit(nopython=True,cache=True)
def circumcenter_periodic(xt,Lx,Ly):
    """
    Find the circumcentre (i.e. vertex position) of each triangle in the triangulation.

    :param C: Cell centroids for each triangle in triangulation (n_c x 3 x 2) np.float32 array
    :param L: Domain size (np.float32)
    :return: Circumcentres/vertex-positions (n_v x 2) np.float32 array
    """

    # disp = (xt[:,0] + xt[:,1] + xt[:,2])/3 - np.array((Lx,Ly/2))
    # ri,rj,rk = mod3(xt-np.expand_dims(disp,1),Lx,Ly).transpose(1,2,0)

    ri,rj,rk = xt.transpose(1,0,2)
    rk0 = rk.copy()
    ri,rj = per(ri-rk,Lx,Ly),per(rj-rk,Lx,Ly)
    rk *=0
    ax, ay = ri.T
    bx, by = rj.T
    cx, cy = rk.T
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (
            ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (
            bx - ax)) / d
    vs = np.empty((ax.size,2),dtype=np.float64)
    vs[:,0],vs[:,1] = ux,uy
    vs = mod2(vs+rk0,Lx,Ly)
    return vs



def triangulate_periodic(x,Lx,Ly):
    """
    Calculates the periodic triangulation on the set of points x.
    :param x: (nc x 2) matrix with the coordinates of each cell
    """

    # 1. Tile cell positions 9-fold to perform the periodic triangulation
    #   Calculates y from x. y is (9nc x 2) matrix, where the first (nc x 2) are the "true" cell positions,
    #   and the rest are translations

    grid_x, grid_y = np.mgrid[-1:2, -1:2]
    grid_x[0, 0], grid_x[1, 1] = grid_x[1, 1], grid_x[0, 0]
    grid_y[0, 0], grid_y[1, 1] = grid_y[1, 1], grid_y[0, 0]
    grid_xy = np.array([grid_x.ravel(), grid_y.ravel()]).T
    grid_xy[:,0] *=Lx
    grid_xy[:,1] *=Ly

    y = make_y(x, grid_xy)


    # 2. Perform the triangulation on y
    #   The **triangle** package (tr) returns a dictionary, containing the triangulation.
    #   This triangulation is extracted and saved as tri
    t = tr.triangulate({"vertices": y})
    tri = t["triangles"]

    # Del = Delaunay(y)
    # tri = Del.simplices
    n_c = x.shape[0]

    # 3. Find triangles with **at least one** cell within the "true" frame (i.e. with **at least one** "normal cell")
    #   (Ignore entries with -1, a quirk of the **triangle** package, which denotes boundary triangles
    #   Generate a mask -- one_in -- that considers such triangles
    #   Save the new triangulation by applying the mask -- new_tri
    tri = tri[(tri != -1).all(axis=1)]
    one_in = (tri < n_c).any(axis=1)
    new_tri = tri[one_in]

    # 4. Remove repeats in new_tri
    #   new_tri contains repeats of the same cells, i.e. in cases where triangles straddle a boundary
    #   Use remove_repeats function to remove these. Repeats are flagged up as entries with the same trio of
    #   cell ids, which are transformed by the mod function to account for periodicity. See function for more details
    n_tri = remove_repeats(new_tri, n_c)
    xt = x[n_tri]
    vs = circumcenter_periodic(xt,Lx,Ly)
    return n_tri,vs



@jit(nopython=True)
def get_neighbours(tri,neigh=None):
    """
    Given a triangulation, find the neighbouring triangles of each triangle.

    By convention, the column i in the output -- neigh -- corresponds to the triangle that is opposite the cell i in that triangle.

    Can supply neigh, meaning the algorithm only fills in gaps (-1 entries)

    :param tri: Triangulation (n_v x 3) np.int32 array
    :param neigh: neighbourhood matrix to update {Optional}
    :return: (n_v x 3) np.int32 array, storing the three neighbouring triangles. Values correspond to the row numbers of tri
    """
    n_v = tri.shape[0]
    if neigh is None:
        neigh = np.ones_like(tri,dtype=np.int32)*-1
    tri_compare = np.concatenate((tri.T, tri.T)).T.reshape((-1, 3, 2))
    for j in range(n_v):#range(n_v):
        tri_sample_flip = np.flip(tri[j])
        tri_i = np.concatenate((tri_sample_flip,tri_sample_flip)).reshape(3,2)
        for k in range(3):
            if neigh[j,k]==-1:
                neighb,l = np.nonzero((tri_compare[:,:,0]==tri_i[k,0])*(tri_compare[:,:,1]==tri_i[k,1]))
                neighb,l = neighb[0],l[0]
                neigh[j,k] = neighb
                neigh[neighb,np.mod(2-l,3)] = j
    return neigh


@jit(nopython=True, cache=True)
def tnorm(x):
    return np.sqrt(x[:, :, 0] ** 2 + x[:, :, 1] ** 2)


@jit(nopython=True, cache=True)
def roll(x, direc=1):
    """
    Jitted equivalent to np.roll(x,-direc,axis=1)

    direc = 1 --> counter-clockwise
    direc = -1 --> clockwise
    :param x:
    :return:
    """
    if direc == -1: #old "roll_forward"
        return np.column_stack((x[:, 2], x[:, :2]))
    elif direc == 1: #old "roll_reverse"
        return np.column_stack((x[:, 1:3], x[:, 0]))

@jit(nopython=True)
def roll3(x,direc=1):
    x_out = np.empty_like(x)
    x_out[:,:,0],x_out[:,:,1] = roll(x[:,:,0],direc=direc),roll(x[:,:,1],direc=direc)
    return x_out

@jit(nopython=True)
def tri_call(val,tri):
    """
    when val has shape (n,3)
    Equiv. to:
    >> val[tri]

    :param val:
    :param tri:
    :return:
    """
    return val.take(tri.ravel()).reshape(-1,3)

@jit(nopython=True)
def tri_call3(val,tri):
    """
    When val has shape (n,3,2)

    Equiv to:
    >> val[tri]

    :param val:
    :param tri:
    :return:
    """
    vali,valj = val[:,0],val[:,1]
    return np.dstack((tri_call(vali,tri),tri_call(valj,tri)))

@jit(nopython=True)
def tri_mat_call(mat,tri,direc=-1):
    """
    If matrix element {i,j} corresponds to the edge value connecting cells i and j,
    then this function returns the edge value connecting a vertex to its counter-clockwise neighbour
    Or equivalently the case where j is CW to i in a given triangle.

    Swap CCW for CW if direc = 1
    :param mat:
    :param tri:
    :param direc:
    :return:
    """
    # return np.dstack((mat[i, j] for (i, j) in zip(tri, roll(tri, direc))))

    nv = tri.shape[0]
    tmat = np.empty((nv,3))
    tri_roll = roll(tri, direc)
    for k in range(nv):
        for m in range(3):
            tri_i,tri_k = tri[k,m],tri_roll[k,m]
            tmat[k,m] = mat[tri_i,tri_k]
    return tmat



def hexagonal_lattice(rows=3, cols=3, noise=0.0005, A=None):
    """
    Assemble a hexagonal lattice

    :param rows: Number of rows in lattice
    :param cols: Number of columns in lattice
    :param noise: Noise added to cell locs (Gaussian SD)
    :return: points (nc x 2) cell coordinates.
    """
    points = []
    for row in range(rows * 2):
        for col in range(cols):
            x = (col + (0.5 * (row % 2))) * np.sqrt(3)
            y = row * 0.5
            x += np.random.normal(0, noise)
            y += np.random.normal(0, noise)
            points.append((x, y))
    points = np.asarray(points)
    if A is not None:
        points = points * np.sqrt(2 * np.sqrt(3) / 3)
    return points


def bounded_hexagonal_lattice(A_target,Lx,Ly,noise=0.005):
    pts = hexagonal_lattice(int(2*Lx/np.sqrt(A_target)),int(2*Lx/np.sqrt(A_target)),noise = noise,A=A_target)
    pts = pts[pts[:,0]<Lx]
    pts = pts[pts[:, 1] < Ly]
    return pts


@jit(nopython=True)
def get_vn(v,v_CCW,Lx,Ly):
    """
    periodic displacement between vertex and its counter-clockwise neighbour (with respect to the cell in question)
    :param v:
    :param v_neigh:
    :param Lx:
    :param Ly:
    :return:
    """
    vn = np.expand_dims(v,1) - v_CCW
    vn = per3(vn,Lx,Ly)
    return vn

@jit(nopython=True)
def get_vc(v,tcnt,Lx,Ly):
    vc = np.expand_dims(v,1)-tcnt
    return per3(vc,Lx,Ly)

@jit(nopython=True)
def get_vc3(v, tcnt, Lx, Ly):
    vc = v - tcnt
    return per3(vc, Lx, Ly)


def get_edge_jac(vc,vc_CCW):
    """
    The cross-product of the vectors {centroid to vertex} and {centroid to CCW vertex}

    :param vc:
    :param vc_CCW:
    :return:
    """
    return vc[:,:,0]*vc_CCW[:,:,1] - vc[:,:,1]*vc_CCW[:,:,0]

def assemble_tri(tval,tri):
    vals = coo_matrix((tval.ravel(),(tri.ravel(),np.zeros_like(tri.ravel()))),shape=(tri.max()+1,1))
    return vals.toarray()


def assemble_tri3(tval,tri):
    vals = coo_matrix((tval.ravel(),(np.repeat(tri.ravel(),2),np.tile((0,1),tri.size))),shape=(tri.max()+1,2))
    return vals.toarray()


@jit(nopython=True)
def sort_coords(coords,centre,start=None):
    if start is None:
        start_angle = 0
    else:
        start_vec = start-centre
        start_angle = np.arctan2(start_vec[1],start_vec[0])
    ncoords = coords-centre
    return coords[np.mod(np.arctan2(ncoords[:,1],ncoords[:,0])-start_angle,np.pi*2).argsort()]



###Archive

#
# def get_CV_matrix(tri, n_v, n_c):
#     CV_matrix = np.zeros((n_c, n_v, 3))
#     for i in range(3):
#         CV_matrix[tri[:, i], np.arange(n_v), i] = 1
#     return CV_matrix
#
# @jit(nopython=True, cache=True)
# def tri_sum(n_c,CV_matrix,tval):
#     val_sum = np.zeros(n_c)
#     for i in range(3):
#         val_sum += np.asfortranarray(CV_matrix[:, :, i]) @ np.asfortranarray(tval[:,i])
#     return val_sum
#
