import os

import click
from pyart.graph.common import generate_radar_time_begin

from radproc.io import read_h5
from radproc.ml import add_mli, ml_grid
from radproc.visual import plot_ml_boundary_level, plot_detected_ml_bounds
from radproc.tools import source2dict


def _out_help():
    base = ('Output HDF5 file PATH. '
            'Special variables {timestamp} and {site} are available. ')
    return base


@click.command()
@click.option('-i', 'infile', metavar='PATH',
              help='input ODIM HDF5 radar volume', required=True)
@click.option('-o', 'outh5', metavar='PATH', help=_out_help())
@click.option('--png-dir', metavar='DIR', help='optional PNG figure output directory')
def main(infile, outh5, png_dir):
    """melting layer analysis"""
    radar = read_h5(infile)
    add_mli(radar)
    bot, top, lims = ml_grid(radar, resolution=50)
    t = generate_radar_time_begin(radar)
    tstamp = t.strftime('%Y%m%d%H%M')
    site = source2dict(radar.metadata['source'])['NOD']
    topbot = {'top': top, 'bottom': bot}
    for key, value in topbot.items():
        fig, ax = plot_ml_boundary_level(value)
        plot_detected_ml_bounds(radar, lims, ax, boundkey=key)
        ax.set_title('Melting layer '+key)
        figfile = os.path.join(png_dir, f"{tstamp}{site}_{key}.png")
        fig.savefig(figfile)
