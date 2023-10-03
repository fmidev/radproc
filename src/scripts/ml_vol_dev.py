import os

import numpy as np
import pyart
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter
from matplotlib.cm import ScalarMappable

from radproc.visual import canvas
from radproc.io import read_h5
from radproc.ml import add_mli, ml_ppi


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





def field_shape(radar):
    field = list(radar.fields)[0] # any field
    return radar.get_field(0, field).shape


def ml_field(radar, add_field=False):
    include = (2, 3, 4)
    data = []
    for sweep in radar.sweep_number['data']:
        phase = radar.get_field(sweep, 'DBZH', copy=True).astype(int)
        mask = phase.mask.copy()
        phase.fill(0)
        if sweep in include:
            bot, top = ml_ppi(radar, sweep)
            for (i, botgate), (_, topgate) in zip(bot.gate.items(), top.gate.items()):
                if botgate<1 or topgate<1:
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


if __name__ == '__main__':
    datadir = os.path.expanduser('~/data/pvol/')
    fname = os.path.join(datadir, '202206030010_fivih_PVOL.h5')
    radar = read_h5(fname)
    add_mli(radar)
    ml_field(radar, add_field=True)
    #
    cm = ListedColormap([p.color for p in PHASE.values()])
    norm_bins = np.array([p.value for p in PHASE.values()])+0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    norm = BoundaryNorm(norm_bins, len(PHASE), clip=True)
    fmt = FuncFormatter(lambda x, pos: PHASE[list(PHASE.keys())[norm(x)]].long_name)
    kws = dict(sweep=2, resolution='50m')
    zkws = dict(vmin=0, vmax=50, cmap='pyart_HomeyerRainbow')
    fig, ax = canvas(radar, 2, 1, right=0.92)
    display = pyart.graph.RadarMapDisplay(radar)
    display.plot_ppi_map('DBZH', ax=ax[0], title='DBZH', **zkws, **kws)
    display.plot_ppi_map('PCLASS', ax=ax[1], title='PCLASS', cmap=cm, norm=norm,
                         colorbar_flag=False, **kws)
    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cm), format=fmt, ticks=tickz)
