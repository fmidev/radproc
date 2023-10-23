"""melting layer detection using scipy peak utilities"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import median_filter, uniform_filter

from radproc.aliases.fmi import ZH, ZDR, RHOHV, MLI, FLTRD_SUFFIX
from radproc.preprocessing import scale_field
from radproc.math import weighted_median, interp_mba
from radproc.tools import find
from radproc.radar import get_field_df, ppi_altitude
from radproc.filtering import (savgol_series, fltr_rolling_median_thresh,
                               fltr_no_hydrometeors, filter_field,
                               filter_series_skipna, fltr_ignore_head)


class PrecipCategory:
    def __init__(self, value, abbr=None, long_name=None, color=None):
        self.value = value
        self.abbr = abbr
        self.long_name = long_name
        self.color = color


PHASE = {'NONE': PrecipCategory(0, abbr='NONE', long_name='not precipitation',
                                color='black'),
         'RAIN': PrecipCategory(1, abbr='RAIN', long_name='rain',
                                color='springgreen'),
         'WETSNOW': PrecipCategory(2, abbr='WETSNOW', long_name='wet snow',
                                   color='violet'),
         'DRYSNOW': PrecipCategory(3, abbr='DRYSNOW', long_name='dry snow',
                                   color='cornflowerblue'),
         'UNKNOWN': PrecipCategory(6, abbr='UNKNOWN', long_name='unable to determine',
                                   color='yellow'),
         'UNDET': PrecipCategory(7, abbr='UNDET', long_name='not analyzed',
                                 color='gray')}

H_MAX = 4200


def indicator_formula(zdr_scaled, zh_scaled, rho, rho_min=0.86):
    """melting layer indicator basic formula"""
    rho[rho < rho_min] = rho_min # rho lower cap; free param
    mli = (1-rho)*(zdr_scaled+1)*zh_scaled*100
    return mli


def indicator(zdr_scaled, zh_scaled, rho, savgol_args=(35, 3), **kws):
    """Calculate ML indicator."""
    mli = indicator_formula(zdr_scaled, zh_scaled, rho, **kws)
    # TODO: check window_length
    mli = mli.apply(savgol_series, args=savgol_args)
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


def ml_h(mlis, **kws):
    """weighted median ML altitude from ML indicator using peak detection"""
    peaksi, peaks = get_peaks(mlis, **kws)
    if peaks.empty:
        return np.nan
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


def limits_peak(peaksi, altitudes):
    """ML height range from MLI peaks"""
    edges = []
    for ips_label in ('left_ips', 'right_ips'):
        ips = get_peaksi_prop(peaksi, ips_label)
        # TODO: instead of the first, use nearest to avg ml altitude
        ilims = ips.apply(_first_or_nan)
        ilims.name = 'gate'
        lims = ilims.apply(_value_at, args=(altitudes,))
        lims.name = 'height'
        lims = pd.concat([lims, ilims.apply(_roundnan)], axis=1)
        edges.append(lims)
    return tuple(edges)


def ml_limits_raw(mli, ml_max_change=1500, mlh=None, **kws): # free param
    """ML height range from ML indicator"""
    if mlh is None:
        mlh = ml_h(mli)
    if np.isnan(mlh):
        nans = pd.Series(index=mli.columns, data=np.nan)
        return nans, nans
    peaksi, _ = get_peaks(mli, hlim=(mlh-ml_max_change, mlh+ml_max_change),
                              **kws)
    return limits_peak(peaksi, mli.index)


def fltr_ml_limits(limits, rho):
    """filter ml range"""
    lims = []
    for lim in limits:
        lim = fltr_rolling_median_thresh(lim, threshold=800) # free param
        lim = fltr_no_hydrometeors(lim, rho)
        lim = lim.rolling(5, center=True, win_type='triang', min_periods=2).mean()
        lims.append(lim)
    return lims


def ml_limits(mli, rho, **kws):
    """filtered ml bottom and top heights"""
    limdfs = ml_limits_raw(mli, **kws)
    if limdfs[0].isna().all().all():
        return limdfs
    lims = tuple((df.height for df in limdfs))
    # filter based on rel_height sensitivity
    lim05dfs = ml_limits_raw(mli, rel_height=0.5, **kws) # free param
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
    """melting indicator from Radar object"""
    zh_scaled = scale_field(radar, ZH)
    zdr_scaled = scale_field(radar, ZDR+FLTRD_SUFFIX, field_type=ZDR)
    rho = radar.fields[RHOHV+FLTRD_SUFFIX]['data']
    return indicator_formula(zdr_scaled, zh_scaled, rho)


def _add_ml_indicator(radar):
    """Calculate and add ML indicator field to Radar object."""
    mlifield = radar.fields[ZH].copy()
    mlifield['data'] = _ml_indicator(radar)
    mlifield['long_name'] = 'Melting layer indicator'
    radar.add_field(MLI, mlifield, replace_existing=True)


def add_mli(radar):
    """Add filtered melting layer indicator to Radar object."""
    filter_field(radar, ZDR, filterfun=median_filter, size=10, mode='wrap')
    filter_field(radar, RHOHV, filterfun=median_filter, size=10, mode='wrap')
    _add_ml_indicator(radar)
    filter_field(radar, MLI, filterfun=fltr_ignore_head, n=3)
    filter_field(radar, MLI+FLTRD_SUFFIX, filterfun=uniform_filter, size=(9,1), mode='wrap')
    try:
        filter_field(radar, MLI+FLTRD_SUFFIX, filterfun=savgol_filter, filled=True,
                     zgate_kw='window_length', polyorder=3, axis=1)
    except ValueError: # polyorder must be less than window_length.
        pass # TODO: warn


def _edge2cartesian(radar, edge, sweep):
    """Series of (ray, gate, altitude) to cartesian coordinates."""
    xyz = radar.get_gate_x_y_z(sweep)
    ed = edge.dropna()
    xs = xyz[0][ed.index.values, ed.gate.values]/1000
    ys = xyz[1][ed.index.values, ed.gate.values]/1000
    zs = ed.height.values
    return np.array(list(zip(xs, ys))), zs


def _edge_gates(edge, height):
    """Find gate numbers corresponding to given altitudes."""
    gates = edge.apply(lambda h: find(height, h))
    gates.name = 'gate'
    return pd.concat([edge, gates], axis=1)


def ml_ppi(radar, sweep, **kws):
    """smooth melting layer detection from a single sweep"""
    # Not filling mli nans with zeros sometimes causes peak depth misdetection.
    # On the other hand, filling may produce unwanted extra peaks.
    mlidf = get_field_df(radar, sweep, MLI+FLTRD_SUFFIX).fillna(0)
    rhodf = get_field_df(radar, sweep, RHOHV+FLTRD_SUFFIX)
    mlh = np.log(mlidf+1).mean(axis=1).idxmax() # "average ml peak"
    bot, top = ml_limits(mlidf, rhodf, mlh=mlh, **kws)
    if bot.isna().all():
        nanarr = np.full([bot.size, 2], np.nan)
        nans = pd.DataFrame(nanarr, columns=['height', 'gate'])
        return nans, nans
    lims = {'bottom': bot, 'top': top}
    h = ppi_altitude(radar, sweep)
    ml_smooth = dict()
    for limlabel in lims.keys():
        limfh = filter_series_skipna(lims[limlabel], uniform_filter, size=30, mode='wrap')
        limfh.name = 'height'
        ml_smooth[limlabel] = _edge_gates(limfh, h)
    return ml_smooth['bottom'], ml_smooth['top']


def ml_grid(radar, sweeps=(2, 3, 4), interpfun=interp_mba, **kws):
    """melting layer height as a grid from a volume scan"""
    # lower threshold when closer to radar (higher elevation)
    max_h_change = {2: 800, 3: 300, 4: 200} # TODO these are from the sleeve
    xys = dict(bottom=[], top=[])
    zs = dict(bottom=[], top=[])
    v = dict()
    all_lims = {}
    for sweep in sweeps:
        bot, top = ml_ppi(radar, sweep, ml_max_change=max_h_change[sweep])
        if bot.isna().all().all():
            continue
        lims = {'bottom': bot, 'top': top}
        all_lims[sweep] = lims
        for limlabel in lims.keys():
            xy, z = _edge2cartesian(radar, lims[limlabel], sweep)
            xys[limlabel].append(xy)
            zs[limlabel].append(z)
    if len(all_lims) == 0:
        try:
            reso = kws['resolution']
        except KeyError:
            reso = 50
        nans = np.full([reso, reso], np.nan)
        return nans, nans, all_lims
    for limlabel in lims.keys():
        xy = np.concatenate(xys[limlabel])
        z = np.concatenate(zs[limlabel])
        v[limlabel] = interpfun(xy, z, **kws)
    return v['bottom'], v['top'], all_lims


def ml_field(radar, add_field=False):
    include = (2, 3, 4)
    data = []
    for sweep in radar.sweep_number['data']:
        phase = radar.get_field(sweep, 'DBZH', copy=True).astype(int)
        mask = phase.mask.copy()
        phase.fill(0)
        if sweep in include:
            bot, top = ml_ppi(radar, sweep)
            for (i, botgate), (_, topgate) in zip(bot['gate'].items(), top['gate'].items()):
                if botgate<1 or topgate<1 or np.isnan(botgate) or np.isnan(topgate):
                    phase[i, :] = PHASE['UNKNOWN'].value
                    continue
                phase[i, :botgate] = PHASE['RAIN'].value
                phase[i, botgate:topgate] = PHASE['WETSNOW'].value
                phase[i, topgate:] = PHASE['DRYSNOW'].value
        else:
            phase[~mask] = PHASE['UNDET'].value
        phase.mask = mask # restore original mask
        data.append(phase)
    pphase = dict(data=np.ma.concatenate(data))
    if add_field:
        radar.add_field('PCLASS', pphase)
    return pphase
