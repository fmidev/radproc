import os

import click
from pyart.graph.common import generate_radar_time_begin

from radproc.io import read_h5, write_h5
from radproc.ml import add_mli, ml_grid, ml_field
from radproc.visual import plot_ml_boundary_level, plot_detected_ml_bounds
from radproc.tools import source2dict


def _out_help():
    base = ('Output HDF5 file PATH. '
            'Special variables {timestamp} and {site} are available. ')
    return base


def plot_ml(radar, png_dir, tstamp, site):
    bot, top, lims = ml_grid(radar, resolution=50)
    topbot = {'top': top, 'bottom': bot}
    for key, value in topbot.items():
        fig, ax = plot_ml_boundary_level(value)
        plot_detected_ml_bounds(radar, lims, ax, boundkey=key)
        ax.set_title('Melting layer '+key)
        figfile = os.path.join(png_dir, f"{tstamp}{site}_{key}.png")
        fig.savefig(figfile)


@click.command()
@click.argument('inputfile', type=click.Path(exists=True, dir_okay=False,
                                             readable=True))
@click.option('-o', '--h5-out', metavar='PATH', help=_out_help())
@click.option('--png-dir', metavar='DIR',
              help='optional PNG figure output directory')
def main(inputfile, h5_out, png_dir):
    """Perform melting layer analysis on INPUTFILE."""
    radar = read_h5(inputfile)
    add_mli(radar)
    t = generate_radar_time_begin(radar)
    tstamp = t.strftime('%Y%m%d%H%M')
    site = source2dict(radar.metadata['source'])['NOD']
    if h5_out:
        ml_field(radar, add_field=True)
        h5out = h5_out.format(timestamp=tstamp, site=site)
        write_h5(radar, h5out, inputfile=inputfile)
    if png_dir:
        plot_ml(radar, png_dir, tstamp, site)
