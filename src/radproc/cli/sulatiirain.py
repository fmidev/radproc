import os

import click
from pyart.graph.common import generate_radar_time_begin
import numpy as np
import pyart
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs

from radproc.visual import canvas
from radproc.radar import altitude_ring
from radproc.io import read_h5, write_h5, read_odim_ml
from radproc.ml import add_mli, ml_grid, ml_field, PHASE
from radproc.visual import plot_ml_boundary_level, plot_detected_ml_bounds
from radproc.tools import source2dict
from radproc._version import __version__


def _out_help():
    base = ('Output HDF5 file PATH. '
            'Special variables {timestamp} and {site} are available. ')
    return base


def plot_analysis(radar, sweep, zerolevel=-1):
    cm = ListedColormap([p.color for p in PHASE.values()])
    norm_bins = np.array([p.value for p in PHASE.values()])+0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    norm = BoundaryNorm(norm_bins, len(PHASE), clip=True)
    fmt = FuncFormatter(lambda x, pos: PHASE[list(PHASE.keys())[norm(x)]].long_name)
    kws = dict(sweep=sweep, resolution='50m')
    zkws = dict(vmin=0, vmax=50, cmap='pyart_HomeyerRainbow')
    fig, ax = canvas(radar, 2, 2, right=0.92)
    display = pyart.graph.RadarMapDisplay(radar)
    display.plot_ppi_map('DBZH', ax=ax[0,0], title='DBZH', **zkws, **kws)
    display.plot_ppi_map('PCLASS', ax=ax[0,1], title='PCLASS', cmap=cm, norm=norm,
                         colorbar_flag=False, **kws)
    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cm), ax=ax[0,1], format=fmt, ticks=tickz)
    display.plot_ppi_map('MLI', ax=ax[1,0], vmin=0, vmax=10, title='MLI', **kws)
    display.plot_ppi_map('MLIC', ax=ax[1,1], vmin=0, vmax=10, title='MLIC', **kws)
    if zerolevel>0:
        lat, lon = altitude_ring(radar, sweep, zerolevel)
        for axx in ax.flatten():
            zoom = 0.9
            xlim = np.array(axx.get_xlim())*zoom
            axx.set_xlim(*xlim)
            ylim = np.array(axx.get_ylim())*zoom
            axx.set_ylim(*ylim)
            axx.plot(lon, lat, transform=ccrs.Geodetic(), color='xkcd:eggplant',
                     linewidth=0.8, label='how/freeze')
    return fig, ax


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
@click.option('--analysis-plot', help='visualize analysis for SWEEP',
              metavar='SWEEP', default=0)
@click.option('--png-dir', metavar='DIR',
              help='optional PNG figure output directory')
@click.version_option(version=__version__, prog_name='sulatiirain')
def main(inputfile, h5_out, png_dir, analysis_plot):
    """Perform melting layer analysis on INPUTFILE."""
    radar = read_h5(inputfile)
    zerolevel = read_odim_ml(inputfile)
    ml_guess = zerolevel-250 # z meters below 0C-level
    add_mli(radar)
    t = generate_radar_time_begin(radar)
    tstamp = t.strftime('%Y%m%d%H%M')
    site = source2dict(radar.metadata['source'])['NOD']
    if h5_out:
        ml_field(radar, add_field=True, mlh=ml_guess)
        h5out = h5_out.format(timestamp=tstamp, site=site)
        write_h5(radar, h5out, inputfile=inputfile)
    if png_dir:
        plot_ml(radar, png_dir, tstamp, site)
    if analysis_plot:
        ml_field(radar, add_field=True, mlh=ml_guess)
        fig, axarr = plot_analysis(radar, analysis_plot, zerolevel)
        plt.show()
