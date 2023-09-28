"""visualizing ml indicator and limits detection"""
import os

import numpy as np
import matplotlib.pyplot as plt

from radproc.aliases.fmi import RHOHV, MLI, FLTRD_SUFFIX
from radproc.visual import plot_ppi, plot_edge
from radproc.io import read_h5
from radproc.ml import add_mli, ml_ppi


if __name__ == '__main__':
    sweep = 3
    ray = 90
    plt.close('all')
    datadir = os.path.expanduser('~/data/pvol/')
    # melting at 1-2km
    f_melt1 = os.path.join(datadir, '202206030010_fivih_PVOL.h5')
    r_melt1 = read_h5(f_melt1)
    add_mli(r_melt1)
    #
    axrho = plot_ppi(r_melt1, vmin=0.86, vmax=1, sweep=sweep, what=RHOHV)
    axrhof = plot_ppi(r_melt1, vmin=0.86, vmax=1, sweep=sweep, what=RHOHV+FLTRD_SUFFIX)
    axmli = plot_ppi(r_melt1, vmin=0, vmax=10, sweep=sweep, what=MLI)
    axf = plot_ppi(r_melt1, vmin=0, vmax=10, sweep=sweep, what=MLI+FLTRD_SUFFIX)
    #
    bot, top = ml_ppi(r_melt1, sweep, ml_max_change=800)
    #
    plot_edge(r_melt1, sweep, bot, axf, color='red')
    plot_edge(r_melt1, sweep, top, axf, color='black')
    plot_edge(r_melt1, sweep, bot, axrho, color='blue')
    plot_edge(r_melt1, sweep, top, axrho, color='black')
    #
    raw = r_melt1.get_field(sweep, MLI)[ray]
    final = r_melt1.get_field(sweep, MLI+FLTRD_SUFFIX)[ray]
    x = np.arange(final.size)
    fig1, ax1 = plt.subplots()
    ax1.plot(x, raw, label='raw')
    ax1.plot(x, final, label='filtered')
    ax1.set_xlim(left=-1, right=200)
