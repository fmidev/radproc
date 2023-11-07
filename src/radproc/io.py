"""reading and writing data"""
import h5py
import numpy as np
import pyart.aux_io
from pyart.core import Radar


def read_odim_ml(h5file):
    """Read how/freeze in meters from FMI ODIM HDF5 radar metadata."""
    with h5py.File(h5file) as f:
        h = f['how'].attrs['freeze']*1000
        if np.isscalar(h):
            return h
        return h.flatten()[0]


def read_h5(filename: str, exclude_datasets=['dataset13'], file_field_names=True,
            **kws) -> Radar:
    """pyart read_odim_h5 wrapper

    Currently, pyart only supports uniform binsize for odim h5.
    Exclude elevations with different binsize."""
    opts = (['dataset7', 'dataset8', 'dataset9'],
            ['dataset7', 'dataset8', 'dataset9', 'dataset10'])
    try:
        return pyart.aux_io.read_odim_h5(filename,
                                         exclude_datasets=exclude_datasets,
                                         file_field_names=file_field_names,
                                         **kws)
    except ValueError:
        # second guesses
        for opt in opts:
            try:
                return pyart.aux_io.read_odim_h5(filename,
                                                 file_field_names=file_field_names,
                                                 exclude_datasets=opt, **kws)
            except ValueError:
                continue


def write_h5(radar: Radar, outfile: str, inputfile='', **kws):
    """pyart write_odim_h5 wrapper"""
    pyart.aux_io.write_odim_h5(outfile, radar, **kws)
    if inputfile:
        with h5py.File(outfile, 'a') as new:
            with h5py.File(inputfile) as old:
                new['how'].attrs.update(old['how'].attrs)
