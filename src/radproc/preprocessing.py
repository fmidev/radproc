# coding: utf-8
""""radar data scaling for classification"""

from sklearn import preprocessing

from radproc.aliases.fmi import ZH, ZDR, KDP


SCALING_LIMITS_SNOW = {ZH: (-10, 34), ZDR: (0, 3.3), KDP: (0, 0.11)}
SCALING_LIMITS_RAIN = {ZH: (-10, 38), ZDR: (0, 3.1), KDP: (0, 0.25)}


def scale(data, param=ZH, has_ml=True, inverse=False):
    """radar data scaling"""
    scaled = data.copy()
    limits = SCALING_LIMITS_RAIN if has_ml else SCALING_LIMITS_SNOW
    if inverse:
        scaled *= limits[param][1]
        scaled += limits[param][0]
    else:
        scaled -= limits[param][0]
        scaled *= 1.0/limits[param][1]
    return scaled


def scale_field(radar, field, field_type=None, **kws):
    """Scale radar field values using RadarDataScaler."""
    if field_type is None:
        field_type=field
    copy = radar.fields[field]['data'].copy() # to be scaled
    scaler = RadarDataScaler(field_type, **kws)
    scaled = scaler.fit_transform(copy)
    return scaled


class RadarDataScaler(preprocessing.FunctionTransformer):
    """FunctionTransformer wrapper"""

    def __init__(self, param=ZH, has_ml=True, **kws):
        self.param = param
        self.has_ml = has_ml
        fun_kws = dict(param=param, has_ml=has_ml, inverse=False)
        inv_kws = dict(param=param, has_ml=has_ml, inverse=True)
        super().__init__(func=scale, inverse_func=scale,
                         kw_args=fun_kws, inv_kw_args=inv_kws, **kws)
