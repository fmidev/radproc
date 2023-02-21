"""melting layer detection using scipy peak utilities"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import median_filter, uniform_filter

from radproc.aliases import zh, zdr, rhohv, mli
from radproc.preprocessing import scale_field
from radproc.math import weighted_median
from radproc import filtering


H_MAX = 4200


def find(arr, value):
    """find closest value using argmin"""
    return abs(arr-value).argmin()


def ind(zdr_scaled, zh_scaled, rho, rho_min=0.86):
    rho[rho < rho_min] = rho_min # rho lower cap; free param
    mli = (1-rho)*(zdr_scaled+1)*zh_scaled*100
    return mli


def indicator(zdr_scaled, zh_scaled, rho, savgol_args=(35, 3), **kws):
    """Calculate ML indicator."""
    mli = ind(zdr_scaled, zh_scaled, rho, **kws)
    # TODO: check window_length
    mli = mli.apply(filtering.savgol_series, args=savgol_args)
    return mli


def get_peaksi_prop(peaksi, prop):
    """return property from peaksi series"""
    return peaksi.apply(lambda x: x[1][prop])


def peak_weights(peaksi):
    """calculate peak weights as prominence*peak_height"""
    heights = get_peaksi_prop(peaksi, 'peak_heights')
    prom = get_peaksi_prop(peaksi, 'prominences')
    return prom*heights


def ml_height_median(peaksi, peaks):
    """weighted median ML height from peak data"""
    weights = peak_weights(peaksi)
    warr = np.sqrt(np.concatenate(weights.values))
    parr = np.concatenate(peaks.values)
    return weighted_median(parr, warr)


def peak_series(s, ilim=(None, None), **kws):
    """scipy peak detection for Series objects"""
    ind, props = find_peaks(s, **kws)
    imin, imax = ilim
    up_sel, low_sel = tuple(np.ones(ind.shape).astype(bool) for i in range(2))
    if imin is not None:
        low_sel = ind > imin
    if imax is not None:
        up_sel = ind < imax
    selection = up_sel & low_sel
    for key in props:
        props[key] = props[key][selection]
    return ind[selection], props


def get_peaks(mli, hlim=(0, H_MAX), height=2, width=0, distance=20,
              prominence=0.3, rel_height=0.6): # free params
    """Apply peak detection to ML indicator."""
    limits = [find(mli.index, lim) for lim in hlim]
    peaksi = mli.apply(peak_series, ilim=limits, height=height, width=width,
                       distance=distance, prominence=prominence,
                       rel_height=rel_height)
    return peaksi, peaksi.apply(lambda i: list(mli.iloc[i[0]].index))


def ml_height(mlis, **kws):
    """weighted median ML height from ML indicator using peak detection"""
    peaksi, peaks = get_peaks(mlis, **kws)
    return ml_height_median(peaksi, peaks)


def _first_or_nan(l):
    """Get first item in iterable if exists, else nan"""
    try:
        return l[0]
    except IndexError:
        return np.nan


def _value_at(ind, values):
    """round index and return corresponding value, nan on ValueError"""
    try:
        return values[round(ind)]
    except ValueError:
        return np.nan


def _roundnan(ind, fill_value=-1):
    """round with fill_value on ValueError"""
    try:
        return round(ind)
    except ValueError:
        return fill_value


def limits_peak(peaksi, heights):
    """ML height range from MLI peaks"""
    edges = []
    for ips_label in ('left_ips', 'right_ips'):
        ips = get_peaksi_prop(peaksi, ips_label)
        ilims = ips.apply(_first_or_nan)
        ilims.name = 'gate'
        lims = ilims.apply(_value_at, args=(heights,))
        lims.name = 'height'
        lims = pd.concat([lims, ilims.apply(_roundnan)], axis=1)
        edges.append(lims)
    return tuple(edges)


def ml_limits_raw(mli, ml_max_change=1500, **kws): # free param
    """ML height range from ML indicator"""
    mlh = ml_height(mli)
    peaksi, _ = get_peaks(mli, hlim=(mlh-ml_max_change, mlh+ml_max_change),
                              **kws)
    return limits_peak(peaksi, mli.index)


def fltr_ml_limits(limits, rho):
    """filter ml range"""
    lims = []
    for lim in limits:
        lim = filtering.fltr_rolling_median_thresh(lim, threshold=800) # free param
        lim = filtering.fltr_no_hydrometeors(lim, rho)
        lim = lim.rolling(5, center=True, win_type='triang', min_periods=2).mean()
        lims.append(lim)
    return lims


def ml_limits(mli, rho, **kws):
    """filtered ml bottom and top heights"""
    limdfs = ml_limits_raw(mli, **kws)
    lims = tuple((df.height for df in limdfs))
    # filter based on rel_height sensitivity
    lim05dfs = ml_limits_raw(mli, rel_height=0.5) # free param
    lims05 = tuple((df.height for df in lim05dfs))
    for lim, lim05 in zip(lims, lims05):
        lim[abs(lim-lim05) > 800] = np.nan # free param
    return fltr_ml_limits(lims, rho)


def hseries2mask(hseries, hindex):
    """boolean mask DataFrame with False below given height limits"""
    return hseries.apply(lambda x: pd.Series(data=hindex > x, index=hindex)).T


def collapse(s_filled_masked):
    """reset ground level according to mask"""
    return s_filled_masked.shift(-s_filled_masked.isnull().sum())


def collapse2top(df_filled, top):
    """Reset ground of a filled DataFrame to specified levels"""
    if df_filled.isnull().any().any():
        raise ValueError('df_filled must not contain NaNs')
    mask = hseries2mask(top.interpolate().dropna(), df_filled.index)
    return df_filled[mask].apply(collapse)


def _ml_indicator(radar):
    zh_scaled = scale_field(radar, zh)
    zdr_scaled = scale_field(radar, zdr+filtering.FLTRD_SUFFIX, field_type=zdr)
    rho = radar.fields[rhohv+filtering.FLTRD_SUFFIX]['data']
    return ind(zdr_scaled, zh_scaled, rho)


def _add_ml_indicator(radar):
    """Calculate and add ML indicator field to Radar object."""
    mlifield = radar.fields[zh].copy()
    mlifield['data'] = _ml_indicator(radar)
    mlifield['long_name'] = 'Melting layer indicator'
    mlifield['coordinates'] = radar.fields[zdr]['coordinates']
    radar.add_field(mli, mlifield, replace_existing=True)


def add_mli(radar):
    """Add filtered melting layer indicator to Radar object."""
    filtering.filter_field(radar, zdr, filterfun=median_filter, size=10, mode='wrap')
    filtering.filter_field(radar, rhohv, filterfun=median_filter, size=10, mode='wrap')
    _add_ml_indicator(radar)
    filtering.filter_field(radar, mli, filterfun=uniform_filter, size=(30,1), mode='wrap')
    filtering.filter_field(radar, mli, filterfun=savgol_filter, window_length=60, polyorder=3, axis=1)
