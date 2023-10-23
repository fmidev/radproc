"""misc tools for working with pyart Radar objects"""
import pandas as pd
import numpy as np
import pyart

from radproc.qpe import rainrate
from radproc.aliases.fmi import ZH, LWE
from radproc.math import db2lin


PYART_AEQD_FMT = '+proj={proj} +lon_0={lon_0} +lat_0={lat_0} +R={R}'


def ppi_altitude(radar, sweep):
    """1D altitude vector along ray from PPI"""
    return radar.get_gate_lat_lon_alt(sweep)[2][1]


def get_field_df(radar, sweep, fieldname):
    """radar field as DataFrame"""
    field = radar.get_field(sweep, fieldname)
    return pd.DataFrame(field.T, index=ppi_altitude(radar, sweep))


def zgates_per_sweep(radar, zlim=650):
    """how many gates until reaching zlim meters above radar for each sweep"""
    sweeps = radar.sweep_number['data']
    return [(radar.get_gate_x_y_z(n)[2][0]<zlim).sum() for n in sweeps]


def z_r_qpe(radar, dbz_field=ZH, lwe_field=LWE, add_field=True):
    """Add precipitation rate field to radar using r(z) relation."""
    dbz = radar.get_field(0, dbz_field)
    z = db2lin(dbz)
    r = rainrate(z)
    rfield = {'units': 'mm h-1', 'data': r}
    if add_field:
        radar.add_field(lwe_field, rfield)
    return rfield


def altitude_ring(radar, sweep, altitude):
    """coordinates of a constant altitude ring for a given sweep"""
    lat, lon, alt = radar.get_gate_lat_lon_alt(sweep)
    idx = np.searchsorted(alt[0], altitude)
    return lat[:, idx], lon[:, idx]


def pyart_aeqd(radar):
    """radar default projection definition as dictionary"""
    lat = radar.latitude['data'][0]
    lon = radar.longitude['data'][0]
    if isinstance(lat, np.ndarray):
        lat = lat[0]
        lon = lon[0]
    return dict(proj='aeqd', lat_0=lat, lon_0=lon, R=6370997)


def dummy_radar(odimfile, include_fields=['DBZH']):
    """Read minimal data to create a dummy radar object."""
    return pyart.aux_io.read_odim_h5(odimfile, include_datasets=['dataset1'],
                                     file_field_names=True,
                                     include_fields=include_fields)
