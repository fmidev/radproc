import os

import click
from pyart.graph.common import generate_radar_time_begin

from radproc.io import read_h5
from radproc.ml import add_mli, ml_grid
from radproc.visual import plot_ml_boundary_level, plot_detected_ml_bounds


@click.command()
@click.option('-i', 'h5file', help='input ODIM HDF5 radar volume', required=True)
@click.argument('outdir', nargs=1)
def main(h5file, outdir):
    radar = read_h5(h5file)
    add_mli(radar)
    bot, top, lims = ml_grid(radar, resolution=50)
    t = generate_radar_time_begin(radar)
    tstamp = t.strftime('%Y%m%d%H%M')
    topbot = {'top': top, 'bottom': bot}
    for key, value in topbot.items():
        fig, ax = plot_ml_boundary_level(value)
        plot_detected_ml_bounds(radar, lims, ax, boundkey=key)
        ax.set_title('Melting layer '+key)
        figfile = os.path.join(outdir, tstamp+key+'.png')
        fig.savefig(figfile)
