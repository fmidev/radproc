import os

import pyart
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, uniform_filter
from scipy.signal import savgol_filter, find_peaks

from radproc.aliases import zh, zdr, rhohv, mli
from radproc.preprocessing import RadarDataScaler
from radproc.ml import ind, ml_limits_raw


FLTRD_SUFFIX = '_filtered'


def plot_pseudo_rhi(radar, what='reflectivity_horizontal', direction=270, **kws):
    fig, ax = plt.subplots()
    xsect = pyart.util.cross_section_ppi(radar, [direction])
    display = pyart.graph.RadarDisplay(xsect)
    display.plot(what, 0, ax=ax, **kws)
    ax.set_ylim(bottom=0, top=11)
    ax.set_xlim(left=0, right=150)
    return ax


def plot_ppi(radar, sweep=0, what='reflectivity_horizontal', **kws):
    fig, ax = plt.subplots()
    ppi = pyart.graph.RadarDisplay(radar)
    ppi.plot(what, sweep, ax=ax, **kws)
    return ax


def read_h5(filename, exclude_datasets=['dataset13'], **kws):
    """pyart read_odim_h5 wrapper"""
    try:
        return pyart.aux_io.read_odim_h5(filename,
                                         exclude_datasets=exclude_datasets,
                                         **kws)
    except ValueError:
        exclude_datasets=['dataset7', 'dataset8', 'dataset9']
        return pyart.aux_io.read_odim_h5(filename,
                                         exclude_datasets=exclude_datasets,
                                         **kws)


def scale_field(radar, field, field_type=None, **kws):
    if field_type is None:
        field_type=field
    copy = radar.fields[field]['data'].copy() # to be scaled
    scaler = RadarDataScaler(field_type, **kws)
    scaled = scaler.fit_transform(copy)
    return scaled


def ml_indicator(radar):
    zh_scaled = scale_field(radar, zh)
    zdr_scaled = scale_field(radar, zdr+FLTRD_SUFFIX, field_type=zdr)
    rho = radar.fields[rhohv+FLTRD_SUFFIX]['data']
    return ind(zdr_scaled, zh_scaled, rho)


def add_ml_indicator(radar):
    mlifield = radar.fields[zh].copy()
    mlifield['data'] = ml_indicator(radar)
    mlifield['long_name'] = 'Melting layer indicator'
    mlifield['coordinates'] = radar.fields[zdr]['coordinates']
    radar.add_field(mli, mlifield, replace_existing=True)


def field_filter(field_data, filterfun=median_filter, **kws):
    filtered = filterfun(field_data, **kws)
    return np.ma.array(filtered, mask=field_data.mask)


def filter_field(radar, fieldname, **kws):
    sweeps = radar.sweep_number['data']
    filtered = np.concatenate([field_filter(radar.get_field(n, fieldname), **kws) for n in sweeps])
    filtered = np.ma.array(filtered, mask=radar.fields[fieldname]['data'].mask)
    if fieldname[-len(FLTRD_SUFFIX):] == FLTRD_SUFFIX:
        fieldname_out = fieldname
    else:
        fieldname_out = fieldname+FLTRD_SUFFIX
    radar.add_field_like(fieldname, fieldname_out, filtered, replace_existing=True)


if __name__ == '__main__':
    plt.close('all')
    datadir = os.path.expanduser('~/data/pvol/')
    f_nomelt1 = os.path.join(datadir, '202302051845_fivih_PVOL.h5')
    # melting at 1-2km
    f_melt1 = os.path.join(datadir, '202206030010_fivih_PVOL.h5')
    r_nomelt1 = read_h5(f_nomelt1)
    r_melt1 = read_h5(f_melt1)
    #ax0 = plot_pseudo_rhi(r_melt1, what='cross_correlation_ratio', direction=90)
    axzh = plot_ppi(r_melt1, sweep=2, what=zh, title_flag=False)
    axrho = plot_ppi(r_melt1, vmin=0.86, vmax=1, sweep=2, what=rhohv, title_flag=False)
    axzdr = plot_ppi(r_melt1, sweep=2, what=zdr, title_flag=False)
    filter_field(r_melt1, zdr, filterfun=median_filter, size=10, mode='wrap')
    filter_field(r_melt1, rhohv, filterfun=median_filter, size=10, mode='wrap')
    axzdrf = plot_ppi(r_melt1, sweep=2, what=zdr+FLTRD_SUFFIX, title_flag=False)
    axrhof = plot_ppi(r_melt1, vmin=0.86, vmax=1, sweep=2, what=rhohv+FLTRD_SUFFIX, title_flag=False)
    add_ml_indicator(r_melt1)
    ax2 = plot_ppi(r_melt1, vmin=0, vmax=10, sweep=2, what=mli, title_flag=False)
    ax3 = plot_pseudo_rhi(r_melt1, vmin=0, vmax=10, what=mli)
    filter_field(r_melt1, mli, filterfun=uniform_filter, size=(9,1), mode='wrap')
    filter_field(r_melt1, mli, filterfun=savgol_filter, window_length=60, polyorder=3, axis=1)
    axf = plot_ppi(r_melt1, vmin=0, vmax=10, sweep=2, what=mli+FLTRD_SUFFIX, title_flag=False)

    mlifd = r_melt1.get_field(2, mli+FLTRD_SUFFIX)
    df = pd.DataFrame(mlifd.T, index=r_melt1.get_gate_lat_lon_alt(2)[2][1])
    bot, top = ml_limits_raw(df)
