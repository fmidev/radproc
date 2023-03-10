"""melting layer detection development script"""
import os

import matplotlib.pyplot as plt

from radproc.aliases import zh, zdr, rhohv, mli
from radproc.visual import (plot_pseudo_rhi, plot_ppi, plot_edge,
                            plot_ml_boundary_level, plot_detected_ml_bounds)
from radproc.filtering import FLTRD_SUFFIX
from radproc.io import read_h5
from radproc.ml import add_mli, ml_grid


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
    vbot, vtop, lims = ml_grid(r_melt1, resolution=50)
    #
    plot_edge(r_melt1, sweep, lims[sweep]['bottom'], axf, color='red')
    plot_edge(r_melt1, sweep, lims[sweep]['top'], axf, color='black')
    plot_edge(r_melt1, sweep, lims[sweep]['bottom'], axrho, color='blue')
    plot_edge(r_melt1, sweep, lims[sweep]['top'], axrho, color='black')
    #
    figv, axv = plot_ml_boundary_level(vtop)
    plot_detected_ml_bounds(r_melt1, lims, axv, boundkey='top')
