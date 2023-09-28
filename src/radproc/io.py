"""reading and writing data"""
import pyart.aux_io


def read_h5(filename, exclude_datasets=['dataset13'], file_field_names=True, **kws):
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
