"""tools utilizing pysteps"""

import numpy as np
import pyart
import pysteps
from radproc.radar import z_r_qpe, dummy_radar, pyart_aeqd, PYART_AEQD_FMT
from scipy.ndimage import map_coordinates


def advection_correction(p1, p2, tdelta=5, dt=1):
    """
    p1, p2: precipitation fields
    tdelta: time between two observations (5 min)
    dt: interpolation timestep (1 min)

    Adopted from pysteps documentation.
    """
    # Evaluate advection
    pp = np.nan_to_num(np.array([p1, p2]))
    oflow_method = pysteps.motion.get_method("LK")
    fd_kwargs = {"buffer_mask": 10}  # avoid edge effects
    flow = oflow_method(np.log(pp), fd_kwargs=fd_kwargs)
    # Perform temporal interpolation
    ppd = np.zeros((pp[0].shape))
    x, y = np.meshgrid(np.arange(pp[0].shape[1], dtype=float),
                       np.arange(pp[0].shape[0], dtype=float))
    for i in range(dt, tdelta + dt, dt):
        pos1 = (y - i/tdelta*flow[1], x - i/tdelta*flow[0])
        pp1 = map_coordinates(pp[0], pos1, order=1)
        pos2 = (y + (tdelta-i)/tdelta*flow[1], x + (tdelta-i)/tdelta*flow[0])
        pp2 = map_coordinates(pp[1], pos2, order=1)
        ppd += (tdelta-i)*pp1 + i*pp2
    return dt/tdelta**2*ppd


def import_fmi_hdf5(fname):
    """
    Interpret precipitation from HDF5 radar data in pysteps compatible format.
    """
    radar = dummy_radar(fname)
    z_r_qpe(radar, dbz_field='DBZH', lwe_field='RATE')
    size = 512
    resolution = 1000
    r_m = size*resolution/2
    grid_shape = (1, size, size)
    grid_limits = ((0, 5000), (-r_m, r_m), (-r_m, r_m))
    proj = pyart_aeqd(radar)
    projstr = PYART_AEQD_FMT.format(**proj)
    grid = pyart.map.grid_from_radars(radar, grid_shape=grid_shape,
                                      grid_limits=grid_limits, fields=['RATE'],
                                      grid_projection=proj)
    data = np.squeeze(grid.fields['RATE']['data'].filled(np.nan))
    meta = dict(projection=projstr,
                cartesian_unit='m',
                x1=grid.x['data'][0],
                x2=grid.x['data'][-1],
                y1=grid.y['data'][0],
                y2=grid.y['data'][-1],
                xpixelsize=resolution,
                ypixelsize=resolution,
                yorigin='upper',
                institution='Finnish Meteorological Institute',
                accutime=5.0,
                unit='mm/h',
                transform=None,
                zerovalue=0.0,
                zr_a=223.0,
                zr_b=1.53)
    return data, meta
