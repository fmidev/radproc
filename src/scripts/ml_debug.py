import os

from radproc.io import read_h5, read_odim_ml
from radproc.ml import add_mli, ml_field


if __name__ == '__main__':
    inputfile = os.path.expanduser('~/data/polar/fivan/201708121530_radar.polar.fivan.h5')
    inputfile = os.path.expanduser('~/data/polar/fikor/202308071655_radar.polar.fikor.h5')
    radar = read_h5(inputfile)
    zerolevel = read_odim_ml(inputfile)
    add_mli(radar)
    ml_field(radar, add_field=True, mlh=zerolevel-250)
