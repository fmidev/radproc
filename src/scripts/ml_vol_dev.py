import os

import numpy as np
import pyart
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs

from radproc.visual import canvas
from radproc.io import read_h5, read_odim_ml
from radproc.ml import add_mli, ml_field, PHASE


def field_shape(radar):
    field = list(radar.fields)[0] # any field
    return radar.get_field(0, field).shape


def altitude_ring(radar, sweep, altitude):
    lat, lon, alt = radar.get_gate_lat_lon_alt(sweep)
    idx = np.searchsorted(alt[0], altitude)
    return lat[:, idx], lon[:, idx]


if __name__ == '__main__':
    datadir = os.path.expanduser('~/data/pvol/')
    datadir = os.path.expanduser('~/data/polar/fikor')
    fname = os.path.join(datadir, '202206030010_fivih_PVOL.h5')
    fname = os.path.join(datadir, '202308080500_radar.polar.fikor.h5')
    radar = read_h5(fname)
    zerolevel = read_odim_ml(fname)
    add_mli(radar)
    ml_field(radar, add_field=True)
    #
    sweep = 2
    cm = ListedColormap([p.color for p in PHASE.values()])
    norm_bins = np.array([p.value for p in PHASE.values()])+0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    norm = BoundaryNorm(norm_bins, len(PHASE), clip=True)
    fmt = FuncFormatter(lambda x, pos: PHASE[list(PHASE.keys())[norm(x)]].long_name)
    kws = dict(sweep=sweep, resolution='50m')
    zkws = dict(vmin=0, vmax=50, cmap='pyart_HomeyerRainbow')
    fig, ax = canvas(radar, 2, 1, right=0.92)
    display = pyart.graph.RadarMapDisplay(radar)
    display.plot_ppi_map('DBZH', ax=ax[0], title='DBZH', **zkws, **kws)
    display.plot_ppi_map('PCLASS', ax=ax[1], title='PCLASS', cmap=cm, norm=norm,
                         colorbar_flag=False, **kws)
    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cm), format=fmt, ticks=tickz)
    lat, lon = altitude_ring(radar, sweep, zerolevel)
    for axx in ax:
        axx.plot(lon, lat, transform=ccrs.Geodetic(), color='black', linewidth=0.8)
