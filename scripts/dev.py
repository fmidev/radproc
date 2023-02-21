import os

import pyart
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from mba import mba2

from radproc.aliases import zh, zdr, rhohv, mli
from radproc.visual import plot_pseudo_rhi, plot_ppi, plot_edge
from radproc.filtering import filter_series_skipna, FLTRD_SUFFIX
from radproc.io import read_h5
from radproc.ml import ind, ml_limits, find, add_mli


def interp_mba(xys, zs, m0, lo=-100, hi=100, resolution=50):
    x = get_grid(lo, hi, resolution)
    interp = mba2([lo, lo], [hi, hi], [m0, m0], xys, zs)
    return interp(x)


def get_grid(*args):
    s = np.linspace(*args)
    return np.array(np.meshgrid(s, s)).transpose([1, 2, 0]).copy()


def edge2cartesian(radar, edge, sweep):
    xyz = radar.get_gate_x_y_z(sweep)
    ed = edge.dropna()
    xs = xyz[0][ed.index.values, ed.gate.values]/1000
    ys = xyz[1][ed.index.values, ed.gate.values]/1000
    zs = ed.height.values
    return np.array(list(zip(xs, ys))), zs


def ppi_altitude(radar, sweep):
    """1D altitude vector along ray from PPI"""
    return radar.get_gate_lat_lon_alt(sweep)[2][1]


def get_field_df(radar, sweep, fieldname):
    """radar field as DataFrame"""
    field = radar.get_field(sweep, fieldname)
    return pd.DataFrame(field.T, index=ppi_altitude(radar, sweep))


def edge_gates(edge, height):
    """Find gate numbers corresponding to given altitudes."""
    gates = edge.apply(lambda h: find(height, h))
    gates.name = 'gate'
    return pd.concat([edge, gates], axis=1)


def ml_ppi(radar, sweep):
    mlidf = get_field_df(radar, sweep, mlif)
    rhodf = get_field_df(radar, sweep, rhohv+FLTRD_SUFFIX)
    bot, top = ml_limits(mlidf, rhodf)
    lims = {'bot': bot, 'top': top}
    h = ppi_altitude(r_melt1, sweep)
    ml_smooth = dict()
    for limlabel in 'bot', 'top':
        limfh = filter_series_skipna(lims[limlabel], uniform_filter, size=30, mode='wrap')
        limfh.name = 'height'
        ml_smooth[limlabel] = edge_gates(limfh, h)
    return ml_smooth['bot'], ml_smooth['top']


if __name__ == '__main__':
    sweep = 2
    mlif = mli+FLTRD_SUFFIX
    plt.close('all')
    datadir = os.path.expanduser('~/data/pvol/')
    f_nomelt1 = os.path.join(datadir, '202302051845_fivih_PVOL.h5')
    # melting at 1-2km
    f_melt1 = os.path.join(datadir, '202206030010_fivih_PVOL.h5')
    r_nomelt1 = read_h5(f_nomelt1)
    r_melt1 = read_h5(f_melt1)
    add_mli(r_melt1)
    #
    #ax0 = plot_pseudo_rhi(r_melt1, what='cross_correlation_ratio', direction=90)
    axzh = plot_ppi(r_melt1, sweep=sweep, what=zh)
    axrho = plot_ppi(r_melt1, vmin=0.86, vmax=1, sweep=sweep, what=rhohv)
    axzdr = plot_ppi(r_melt1, sweep=sweep, what=zdr)
    axzdrf = plot_ppi(r_melt1, sweep=sweep, what=zdr+FLTRD_SUFFIX)
    axrhof = plot_ppi(r_melt1, vmin=0.86, vmax=1, sweep=sweep, what=rhohv+FLTRD_SUFFIX)
    ax2 = plot_ppi(r_melt1, vmin=0, vmax=10, sweep=sweep, what=mli)
    ax3 = plot_pseudo_rhi(r_melt1, vmin=0, vmax=10, what=mli)
    axf = plot_ppi(r_melt1, vmin=0, vmax=10, sweep=sweep, what=mlif)
    #
    bot, top = ml_ppi(r_melt1, sweep)
    #
    plot_edge(r_melt1, sweep, bot, axf, color='red')
    plot_edge(r_melt1, sweep, top, axf, color='black')
    plot_edge(r_melt1, sweep, bot, axrho, color='blue')
    plot_edge(r_melt1, sweep, top, axrho, color='black')
    #
    xys, zs = edge2cartesian(r_melt1, top, sweep)
    v = interp_mba(xys, zs, 2, resolution=50)
    s = np.linspace(-100,100)
    figm, axm = plt.subplots()
    axm.pcolormesh(s, s, v)
    plot_edge(r_melt1, sweep, top, axm, color='red')
    axm.axis('equal')
