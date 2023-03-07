"""reading and writing data"""
import pyart.aux_io


def read_h5(filename, exclude_datasets=['dataset13'], **kws):
    """pyart read_odim_h5 wrapper

    Currently, pyart only supports uniform binsize for odim h5.
    Exclude elevations with different binsize."""
    try:
        return pyart.aux_io.read_odim_h5(filename,
                                         exclude_datasets=exclude_datasets,
                                         **kws)
    except ValueError:
        # second guess
        exclude_datasets=['dataset7', 'dataset8', 'dataset9']
        return pyart.aux_io.read_odim_h5(filename,
                                         exclude_datasets=exclude_datasets,
                                         **kws)
