import numpy as np
import matplotlib.pyplot as plt
from tri_functions import *
import os
from matplotlib import animation
import time

@jit(nopython=True)
def get_e_stack(vi,vj):
    e_stack = np.zeros((vi.shape[0],3,2,2))
    for i in range(vi.shape[0]):
        for j in range(3):
            e_stack[i,j] = np.row_stack((vi[i,j],vj[i,j]))
    return e_stack


def plot_mesh(ax,v,neigh,Lx,Ly,color="black",alpha=1):
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



def animate_mesh(v_save,neigh_save,Lx,Ly,n_frames=100, file_name=None, dir_name="plots"):

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    xlim = (0,Lx)
    ylim = (0,Ly)

    skip = int((v_save.shape[0]) / n_frames)

    def animate(i):
        ax1.cla()

        v = v_save[i*skip]
        neigh = neigh_save[i*skip]

        plot_mesh(ax1,v,neigh,Lx,Ly)

        # ax1.set_facecolor('black')

        # ax1.set(aspect=1, xlim=xlim, ylim=ylim)

    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=20, bitrate=500)
    if file_name is None:
        file_name = "animation %d" % time.time()
    an = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200)
    an.save("%s/%s.mp4" % (dir_name, file_name), writer=writer, dpi=264)

