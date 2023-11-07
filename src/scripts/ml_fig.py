"""melting layer detection development script"""
import os
from functools import partial

import matplotlib.pyplot as plt
import pyart

from radproc.io import read_h5
from radproc.ml import add_mli
from radproc.visual import coord_altitude


if __name__ == '__main__':
    sweep = 2
    plt.close('all')
    datadir = os.path.expanduser('~/data/pvol/')
    # melting at 1-2km
    f_melt1 = os.path.join(datadir, '202206030010_fivih_PVOL.h5')
    radar = read_h5(f_melt1)
    add_mli(radar)
    #
    fig, ax = plt.subplots()
    ppi = pyart.graph.RadarDisplay(radar)
    ppi.plot('DBZH', ax=ax, title_flag=False)
    r_km = 100
    ppi.set_aspect_ratio(1, ax=ax)
    ppi.set_limits(xlim=(-r_km, r_km), ylim=(-r_km, r_km), ax=ax)
    ax.format_coord = partial(coord_altitude, radar, sweep)
