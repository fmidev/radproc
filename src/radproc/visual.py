"""visualization"""
import pyart
import matplotlib.pyplot as plt


def plot_pseudo_rhi(radar, what='reflectivity_horizontal', direction=270, **kws):
    """Plot a radial cross section from a volume scan."""
    fig, ax = plt.subplots()
    xsect = pyart.util.cross_section_ppi(radar, [direction])
    display = pyart.graph.RadarDisplay(xsect)
    display.plot(what, 0, ax=ax, **kws)
    ax.set_ylim(bottom=0, top=11)
    ax.set_xlim(left=0, right=150)
    return ax


def plot_ppi(radar, sweep=0, what='reflectivity_horizontal', r_km=120,
             title_flag=False, **kws):
    """Plot a single elevation PPI from a volume scan."""
    fig, ax = plt.subplots()
    ppi = pyart.graph.RadarDisplay(radar)
    ppi.plot(what, sweep, ax=ax, title_flag=title_flag, **kws)
    ax.axis('equal')
    ax.set_xlim(-r_km, r_km)
    ax.set_ylim(-r_km, r_km)
    return ax


def plot_edge(radar, sweep, edge, ax, color='red'):
    """
    Plot Series of (ray, gate) pairs in radar-based cartesian coordinates.
    """
    x = radar.get_gate_x_y_z(sweep)[0][edge.index.values, edge.gate]
    y = radar.get_gate_x_y_z(sweep)[1][edge.index.values, edge.gate]
    ax.scatter(x/1000, y/1000, marker=',', color=color, edgecolors='none', s=2)
