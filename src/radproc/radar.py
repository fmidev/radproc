"""misc tools for working with pyart Radar objects"""
import pandas as pd


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
