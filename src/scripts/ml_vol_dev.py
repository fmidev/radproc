import os

from radproc.io import read_h5
from radproc.ml import add_mli, ml_ppi


PHASE = {'NONE': 0, # not precipitation
         'RAIN': 1,
         'WETSNOW': 2,
         'DRYSNOW': 3,
         'UNKNOWN': 4, # not able to determine
         'UNDET': 7} # no effort to determine


def field_shape(radar):
    field = list(radar.fields)[0] # any field
    return radar.get_field(0, field).shape


if __name__ == '__main__':
    datadir = os.path.expanduser('~/data/pvol/')
    fname = os.path.join(datadir, '202206030010_fivih_PVOL.h5')
    radar = read_h5(fname)
    add_mli(radar)
    bot, top = ml_ppi(radar, 2)
    phase = radar.get_field(2, 'DBZH', copy=True).astype(int)
    mask = phase.mask.copy()
    phase.fill(0)
    for (i, botgate), (_, topgate) in zip(bot.gate.items(), top.gate.items()):
        if botgate<1 or topgate<1:
            phase[i, :] = PHASE['UNKNOWN']
            continue
        phase[i, :botgate] = PHASE['RAIN']
        phase[i, botgate:topgate] = PHASE['WETSNOW']
        phase[i, topgate:] = PHASE['DRYSNOW']
    phase.mask = mask # restore original mask
