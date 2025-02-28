
from datetime import datetime

import cmcrameri.cm as cmc
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from cartopy.io import shapereader
from matplotlib import colors
from scipy.ndimage import zoom
from seismostats.plots.basics import dot_size
from shapely import Polygon
from streamlit_js_eval import streamlit_js_eval

from app.plots.colorscales import lajolla_r


def plot_rel_map(ratio: np.ndarray,
                 scatter_catalog: pd.DataFrame,
                 m_thresh: float,
                 bounding_polygon: Polygon,
                 starttime: datetime) -> plt.Figure:

    cmap = cmc.lajolla_r
    cmap.set_under(cmap(0.0))
    scatter_color = 'k'

    # use logarithmic colorscale
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1)

    # use discrete colormap with 5 colors
    norm_min = 0.5
    norm_max = 3.1
    norm = colors.BoundaryNorm(np.arange(norm_min, norm_max, 0.02), cmap.N)

    ax.set_title('Earthquake forecast for Switzerland on {}'.format(
        starttime.strftime('%d.%m.%Y %H:%M')))

    # plot forecast values
    min_lat, min_lon, max_lat, max_lon = bounding_polygon.bounds
    im = ax.imshow(
        np.log10(ratio),
        origin='lower',
        cmap=cmap,
        norm=norm,
        extent=[min_lon, max_lon, min_lat, max_lat],
        interpolation='bilinear',
    )

    # plot observed events
    dot_sizes = dot_size([*scatter_catalog['magnitude'], 7.5],
                         smallest=10, largest=2600, interpolation_power=3)[:-1]

    ax.scatter(scatter_catalog['longitude'],
               scatter_catalog['latitude'],
               color='none',
               edgecolor=scatter_color,
               marker='o',
               s=dot_sizes,
               linewidth=0.5,
               )

    # show only the part inside the polygon
    x, y = bounding_polygon.exterior.xy
    data_clip_path = list(zip(y, x))
    chpoly = plt.Polygon(data_clip_path,
                         edgecolor='k',
                         facecolor='none',
                         lw=0,
                         zorder=10,
                         transform=ax.transData,
                         figure=fig)
    ax.add_patch(chpoly)
    im.set_clip_path(chpoly)

    # border of Switzerland
    shpfilename = shapereader.natural_earth('10m',
                                            'cultural',
                                            'admin_0_countries')
    df = geopandas.read_file(shpfilename)
    poly = [df.loc[df['ADMIN'] == 'Switzerland']['geometry'].values[0]]

    ax.plot(poly[0].exterior.xy[0],
            poly[0].exterior.xy[1],
            color=scatter_color,
            lw=1,
            zorder=10)

    # Colorbar and legend
    cbar = plt.colorbar(
        im, label=f'Probability increase to a normal day\nof at least one M≥{
            m_thresh:.1f} event in 7 days',
        shrink=0.5, orientation='horizontal', pad=0.0, extend='max')
    cbar.ax.minorticks_off()
    cbar_ticks = np.arange(norm_min, norm_max, 0.5)
    cbar_labels = [r'{}$\times$'.format(np.format_float_positional(
        10**i, trim='-', precision=0)) for i in cbar_ticks]
    if cbar_ticks[0] > 0:
        cbar_labels[0] = '1-' + cbar_labels[0]
    cbar_labels[-1] = '≥' + cbar_labels[-1]
    cbar.set_ticks(cbar_ticks, labels=cbar_labels)

    ax.set_aspect(1.4)
    ax.axis('off')

    return fig


def create_line_plot(data,
                     x_col,
                     y_col,
                     xlabel=None,
                     ylabel=None):
    """
    Creates a line plot from a given DataFrame and returns the figure object.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - x_col (str): Column name for x-axis.
    - y_col (str): Column name for y-axis.
    - title (str, optional): Title of the plot. Default is "Line Plot".
    - xlabel (str, optional): Label for x-axis. Default is None (uses x_col).
    - ylabel (str, optional): Label for y-axis. Default is None (uses y_col).

    Returns:
    - fig (matplotlib.figure.Figure): The created figure object.
    """

    fig, ax = plt.subplots(figsize=(8, 3))  # Create figure and axis
    ax.plot(data[x_col], data[y_col], marker=None,
            linestyle="-", color="b", label=y_col)
    # fix the y axis to start at 0
    ax.set_ylim(bottom=0)
    # Labels and title
    ax.set_xlabel(xlabel if xlabel else x_col)
    ax.set_ylabel(ylabel if ylabel else y_col)

    # Additional styling
    ax.grid(True)

    return fig  # Return the figure object


def plot_rel_map_plotly(ratio: np.ndarray,
                        scatter_catalog: pd.DataFrame,
                        m_thresh: float,
                        bounding_polygon: Polygon,
                        starttime):

    log_ratio = np.log10(ratio)
    norm_min, norm_max = 0.5, 3.1

    # extent
    min_lat, min_lon, max_lat, max_lon = bounding_polygon.bounds

    # Upsample the data to 100x100 for a smoother appearance
    high_res_factor = 10
    log_ratio = zoom(log_ratio,
                     high_res_factor,
                     order=1,
                     grid_mode=True,
                     mode='grid-constant')

    x = np.arange(min_lon, max_lon, (max_lon - min_lon) / log_ratio.shape[1])
    y = np.arange(min_lat, max_lat, (max_lat
                                     - min_lat) / log_ratio.shape[0])

    cbar_ticks = np.arange(norm_min, norm_max, 0.5)
    cbar_labels = [r'{}x'.format(np.format_float_positional(
        10**i, trim='-', precision=0)) for i in cbar_ticks]
    if cbar_ticks[0] > 0:
        cbar_labels[0] = '1-' + cbar_labels[0]
    cbar_labels[-1] = '≥' + cbar_labels[-1]

    cbar_title = 'Probability increase to a normal day<br>' \
        f'of at least one M≥{m_thresh:.1f} event in 7 days'

    # Create figure
    fig = go.Figure(
        data=go.Heatmap(
            z=log_ratio,
            x=x,
            y=y,
            colorscale=lajolla_r,
            zmin=norm_min,
            zmax=norm_max,
            showscale=True,
            colorbar=dict(
                title=cbar_title,
                title_side='bottom',
                tickvals=cbar_ticks,
                ticktext=cbar_labels,
                tickmode='array',
                ticks="outside",
                xpad=0,
                x=0.5,
                yanchor='bottom',
                y=-0.2,
                orientation='h',
                thickness=20,
            )
        ))

    # border of Switzerland
    shpfilename = shapereader.natural_earth('10m',
                                            'cultural',
                                            'admin_0_countries')
    df = geopandas.read_file(shpfilename)
    poly = df.loc[df['ADMIN'] == 'Switzerland']['geometry'].values[0]
    border_x, border_y = poly.exterior.xy

    # Add Switzerland border overlay
    fig.add_trace(go.Scatter(
        x=list(border_x),
        y=np.array(border_y),
        mode="lines",
        showlegend=False,
        line=dict(color="black", width=1.5)
    ))

    # Define a bounding box to act as background
    clip_x, clip_y = list(bounding_polygon.exterior.xy[1]), list(
        bounding_polygon.exterior.xy[0])
    bbox_x = [min_lon, max_lon, max_lon, min_lon, min_lon]
    bbox_y = [min_lat, min_lat, max_lat, max_lat, min_lat]

    fig.add_trace(go.Scatter(
        x=bbox_x + clip_x[::-1],
        y=bbox_y + clip_y[::-1],
        fill="toself",
        mode="none",
        fillcolor="rgba(255,255,255,1)",
        showlegend=False,
    ))

    # plot observed events
    dot_sizes = dot_size([*scatter_catalog['magnitude'], 7.5],
                         smallest=4, largest=300, interpolation_power=3.2)[:-1]

    # Add scatter plot overlay
    fig.add_trace(go.Scatter(
        x=scatter_catalog["longitude"],
        y=scatter_catalog["latitude"],
        mode="markers",
        showlegend=False,
        marker=dict(
            size=dot_sizes,  # Match Matplotlib dot sizes
            color="rgba(0,0,0,0)",  # Transparent fill
            line=dict(color='black', width=0.8),  # Black edge
        )
    ))

    width = streamlit_js_eval(js_expressions="window.innerWidth", key='SCR')
    width = width or 800
    # Layout adjustments
    fig.update_layout(
        autosize=True,
        height=1 / 1.4 * width + 80,
        # dragmode="select",  # Allows box/lasso selection
        xaxis=dict(range=[min_lon, max_lon],
                   showticklabels=False,  # Hide x-axis labels
                   showgrid=False,  # Remove grid
                   zeroline=False,  # Hide x-axis line
                   domain=[0, 1]  # Full width
                   ),

        yaxis=dict(range=[min_lat, max_lat],
                   scaleanchor="x",
                   scaleratio=1.4,
                   showticklabels=False,  # Hide x-axis labels
                   showgrid=False,  # Remove grid
                   zeroline=False,  # Hide x-axis line
                   domain=[0, 1]  # Full height
                   ),
        margin=dict(l=0, r=0, t=0, b=100)
    )

    return fig
