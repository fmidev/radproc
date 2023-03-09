"""melting layer detection development script"""
import os

import numpy as np
import matplotlib.pyplot as plt

from radproc.aliases import zh, zdr, rhohv, mli
from radproc.visual import plot_pseudo_rhi, plot_ppi, plot_edge
from radproc.filtering import FLTRD_SUFFIX
from radproc.io import read_h5
from radproc.ml import add_mli, ml_ppi, ml_grid


if __name__ == '__main__':
    sweep = 2
    plt.close('all')
    datadir = os.path.expanduser('~/data/pvol/')
    f_nomelt1 = os.path.join(datadir, '202302051845_fivih_PVOL.h5')
    # melting at 1-2km
    f_melt1 = os.path.join(datadir, '202206030010_fivih_PVOL.h5')
    r_nomelt1 = read_h5(f_nomelt1)
    r_melt1 = read_h5(f_melt1)
    add_mli(r_melt1)
    #
    axzh = plot_ppi(r_melt1, sweep=sweep, what=zh)
    axrho = plot_ppi(r_melt1, vmin=0.86, vmax=1, sweep=sweep, what=rhohv)
    axzdr = plot_ppi(r_melt1, sweep=sweep, what=zdr)
    axzdrf = plot_ppi(r_melt1, sweep=sweep, what=zdr+FLTRD_SUFFIX)
    axrhof = plot_ppi(r_melt1, vmin=0.86, vmax=1, sweep=sweep, what=rhohv+FLTRD_SUFFIX)
    ax2 = plot_ppi(r_melt1, vmin=0, vmax=10, sweep=sweep, what=mli)
    ax3 = plot_pseudo_rhi(r_melt1, vmin=0, vmax=10, what=mli)
    axf = plot_ppi(r_melt1, vmin=0, vmax=10, sweep=sweep, what=mli+FLTRD_SUFFIX)
    #
    bot, top = ml_ppi(r_melt1, sweep, ml_max_change=800)
    bot3, top3 = ml_ppi(r_melt1, 3, ml_max_change=300)
    bot4, top4 = ml_ppi(r_melt1, 4, ml_max_change=200)
    #
    plot_edge(r_melt1, sweep, bot, axf, color='red')
    plot_edge(r_melt1, sweep, top, axf, color='black')
    plot_edge(r_melt1, sweep, bot, axrho, color='blue')
    plot_edge(r_melt1, sweep, top, axrho, color='black')
    #
    s = np.linspace(-100, 100)
    vbot, vtop = ml_grid(r_melt1, resolution=50)
    figv, axv = plt.subplots()
    mesh = axv.pcolormesh(s, s, vtop, cmap='pyart_RefDiff')
    plot_edge(r_melt1, sweep, top, axv, color='black')
    plot_edge(r_melt1, 3, top3, axv, color='black')
    plot_edge(r_melt1, 4, top4, axv, color='black')
    axv.set_aspect('equal', 'box')
    axv.set_xlabel('East West distance from radar [km]')
    axv.set_ylabel('North South distance from radar [km]')
    cb = figv.colorbar(mesh)
    cb.set_label('Altitude [m]')
