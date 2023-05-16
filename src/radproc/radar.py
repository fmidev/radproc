"""misc tools for working with pyart Radar objects"""
import pandas as pd

from radproc.qpe import rainrate
from radproc.aliases import zh, lwe
from radproc.math import db2lin


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


def source2dict(radar):
    """radar metadata source as dictionary"""
    src = radar.metadata['source']
    l = src.split(',')
    d = {}
    for pair in l:
        key, value = pair.split(':')
        d[key] = value
    return d


def z_r_qpe(radar, dbz_field=zh):
    """Add precipitation rate field to radar using r(z) relation."""
    dbz = radar.get_field(0, dbz_field)
    z = db2lin(dbz)
    r = rainrate(z)
    rfield = {'units': 'mm h-1', 'data': r}
    radar.add_field(lwe, rfield)
