"""simple radar figure script"""
import os
from functools import partial

import matplotlib.pyplot as plt
import pyart
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs

from radproc.visual import canvas
from radproc.radar import altitude_ring
from radproc.io import read_h5, read_odim_ml
from radproc.ml import add_mli, ml_grid, ml_field, PHASE
from radproc.visual import (plot_ml_boundary_level, plot_detected_ml_bounds,
                            coord_altitude)
from radproc.tools import source2dict
from radproc._version import __version__

def plot_analysis(radar, sweep, zerolevel=-1):
    cm = ListedColormap([p.color for p in PHASE.values()])
    norm_bins = np.array([p.value for p in PHASE.values()])+0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    norm = BoundaryNorm(norm_bins, len(PHASE), clip=True)
    fmt = FuncFormatter(lambda x, pos: PHASE[list(PHASE.keys())[norm(x)]].long_name)
    kws = dict(sweep=sweep, resolution='50m')
    zkws = dict(vmin=0, vmax=50, cmap='pyart_HomeyerRainbow')
    fig, ax = canvas(radar, 3, 1, right=0.92)
    display = pyart.graph.RadarMapDisplay(radar)
    display.plot_ppi_map('DBZH', ax=ax[0], title='DBZH', **zkws, **kws)
    display.plot_ppi_map('PCLASS', ax=ax[2], title='PCLASS', cmap=cm, norm=norm,
                         colorbar_flag=False, **kws)
    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cm), ax=ax[2], format=fmt, ticks=tickz)
    display.plot_ppi_map('RHOHV', ax=ax[1], vmin=0.9, vmax=1, title='RHOHV', **kws)
    if zerolevel>0:
        lat, lon = altitude_ring(radar, sweep, zerolevel)
        for axx in ax.flatten():
            zoom = 0.8
            xlim = np.array(axx.get_xlim())*zoom
            axx.set_xlim(*xlim)
            ylim = np.array(axx.get_ylim())*zoom
            axx.set_ylim(*ylim)
            axx.plot(lon, lat, transform=ccrs.Geodetic(), color='xkcd:eggplant',
                     linewidth=0.8, label='how/freeze')
            axx.format_coord = partial(coord_altitude, radar, sweep)
    return fig, ax


if __name__ == '__main__':
    sweep = 2
    plt.close('all')
    datadir = os.path.expanduser('~/data/pvol/')
    # melting at 1-2km
    f_melt1 = os.path.join(datadir, '202206030010_fivih_PVOL.h5')
    radar = read_h5(f_melt1)
    z=read_odim_ml(f_melt1)
    ml_guess = z-250
    add_mli(radar)
    ml_field(radar, add_field=True, mlh=ml_guess)
    #
    fig, ax = plot_analysis(radar, 2, zerolevel=z)
