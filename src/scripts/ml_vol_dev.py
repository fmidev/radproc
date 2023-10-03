import os

import numpy as np
import pyart

from radproc.visual import canvas
from radproc.io import read_h5
from radproc.ml import add_mli, ml_ppi


PHASE = {'NONE': 0, # not precipitation
         'RAIN': 1,
         'WETSNOW': 2,
         'DRYSNOW': 3,
         'UNKNOWN': 6, # not able to determine
         'UNDET': 7} # no effort to determine


def field_shape(radar):
    field = list(radar.fields)[0] # any field
    return radar.get_field(0, field).shape


if __name__ == '__main__':
    datadir = os.path.expanduser('~/data/pvol/')
    fname = os.path.join(datadir, '202206030010_fivih_PVOL.h5')
    radar = read_h5(fname)
    add_mli(radar)
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
                    phase[i, :] = PHASE['UNKNOWN']
                    continue
                phase[i, :botgate] = PHASE['RAIN']
                phase[i, botgate:topgate] = PHASE['WETSNOW']
                phase[i, topgate:] = PHASE['DRYSNOW']
        else:
            phase[~mask] = PHASE['UNDET']
        phase.mask = mask # restore original mask
        data.append(phase)
    pphase = dict(data=np.ma.concatenate(data))
    radar.add_field('PCLASS', pphase)
    kws = dict(sweep=2, resolution='50m')
    fig, ax = canvas(radar, 2, 1)
    display = pyart.graph.RadarMapDisplay(radar)
    display.plot_ppi_map('MLI', ax=ax[0], title='MLI', **kws)
    display.plot_ppi_map('PCLASS', ax=ax[1], title='PCLASS', **kws)
