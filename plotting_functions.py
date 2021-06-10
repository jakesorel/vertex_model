import numpy as np
import matplotlib.pyplot as plt
from tri_functions import *
import os
from matplotlib import animation
import time
from mpl_toolkits.axes_grid1 import Divider, Size

@jit(nopython=True)
def get_e_stack(vi,vj):
    e_stack = np.zeros((vi.shape[0],3,2,2))
    for i in range(vi.shape[0]):
        for j in range(3):
            e_stack[i,j] = np.row_stack((vi[i,j],vj[i,j]))
    return e_stack


def plot_mesh(ax,v,neigh,Lx,Ly,color="black",alpha=1):
    v = v[~np.isnan(v[:,0])]
    neigh = neigh[neigh[:,0]!=-1]
    v_neigh = tri_call3(v, neigh)
    v_CCW = roll3(v_neigh, 1)
    vn = get_vn(v, v_CCW, Lx, Ly)
    vi = np.expand_dims(v,1)*np.ones_like(vn)
    vj = vi-vn
    e_stack = get_e_stack(vi,vj)
    ax.plot(e_stack[:,:,:,0].reshape(-1,2).T,e_stack[:,:,:,1].reshape(-1,2).T,color=color,alpha=alpha)
    # for i in range(vi.shape[0]):
    #     for j in range(3):
    #         e = np.row_stack((vi[i,j],vj[i,j]))
    #         ax.plot(e[:,0],e[:,1],color="black")
    ax.set(xlim=(0,Lx),ylim=(0,Ly),aspect=1)


def plot_tri(ax,tcnt,Lx,Ly,color="black"):
    tcnt = tcnt[~np.isnan(tcnt[:,0])]
    for i in range(3):
        cnts = np.array((tcnt[:, i], tcnt[:, np.mod(i + 1, 3)]))
        cnts[1] = cnts[0] + per(cnts[1]-cnts[0],Lx,Ly)
        ax.plot(cnts[:,:,0],cnts[:,:,1],color=color)

def fix_axes_size_incm(axew, axeh,fig,ax,pad=0.8):
    axew = axew*pad
    axeh = axeh*pad

    #lets use the tight layout function to get a good padding size for our axes labels.
    # fig = plt.gcf()
    # ax = plt.gca()
    # fig.tight_layout()
    #obtain the current ratio values for padding and fix size
    oldw, oldh = fig.get_size_inches()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom

    #work out what the new  ratio values for padding are, and the new fig size.
    neww = axew+oldw*(1-r+l)
    newh = axeh+oldh*(1-t+b)
    newr = r*oldw/neww
    newl = l*oldw/neww
    newt = t*oldh/newh
    newb = b*oldh/newh

    #right(top) padding, fixed axes size, left(bottom) pading
    hori = [Size.Scaled(newr), Size.Fixed(axew), Size.Scaled(newl)]
    vert = [Size.Scaled(newt), Size.Fixed(axeh), Size.Scaled(newb)]

    divider = Divider(fig, (0.0, 0.0, 1., 1.), hori, vert, aspect=False)
    # the width and height of the rectangle is ignored.

    ax.set_axes_locator(divider.new_locator(nx=1, ny=1))

    #we need to resize the figure now, as we have may have made our axes bigger than in.
    fig.set_size_inches(neww,newh)


def animate_mesh(v_save,neigh_save,L,n_frames=100, file_name=None, dir_name="plots",scale=10):
    if L.shape == L.size:
        L = np.ones((v_save.shape[0],1))*np.expand_dims(L,0)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


    fig = plt.figure(figsize=(scale,scale*L[:,1].max()/L[:,0].max()))
    sf = scale/L[:,0].max()
    ax1 = fig.add_subplot(1, 1, 1)


    skip = int((v_save.shape[0]) / n_frames)

    def animate(i):

        ax1.cla()

        v = v_save[i*skip]
        neigh = neigh_save[i*skip]
        Lx,Ly = L[i*skip]
        plot_mesh(ax1,v,neigh,Lx,Ly)
        xlim = (0, Lx)
        ylim = (0, Ly)
        # ax1.set_facecolor('black')


        ax1.set(aspect=1, xlim=xlim, ylim=ylim)
        fix_axes_size_incm(Lx*sf,Ly*sf,fig,ax1)

    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=20, bitrate=500)
    if file_name is None:
        file_name = "animation %d" % time.time()
    an = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200)
    an.save("%s/%s.mp4" % (dir_name, file_name), writer=writer, dpi=264)

