
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import zoom
from seismostats.plots.basics import dot_size
from shapely import Polygon

from app.plots.colorscales import lajolla_r


def plot_rel_map_plotly(ratio: np.ndarray,
                        scatter_catalog: pd.DataFrame,
                        bounding_polygon: Polygon,
                        selection_polygon: Polygon,
                        border_polygon: Polygon,
                        width: float = 800) -> go.Figure:

    log_ratio = np.log10(ratio)
    norm_min, norm_max = 0.5, 3.1

    # extent
    min_lon, min_lat, max_lon, max_lat = bounding_polygon.bounds

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
                tickvals=cbar_ticks,
                ticktext=cbar_labels,
                tickmode='array',
                ticks="outside",
                xpad=0,
                x=0.5,
                yanchor='bottom',
                y=-0.05,
                orientation='h',
                thickness=20,
                len=0.5,
            )
        ))

    # Define a bounding box to act as background
    clip_x, clip_y = selection_polygon.exterior.xy
    bbox_x = [min_lon, max_lon, max_lon, min_lon, min_lon]
    bbox_y = [min_lat, min_lat, max_lat, max_lat, min_lat]

    fig.add_trace(go.Scatter(
        x=bbox_x + list(clip_x[::-1]),
        y=bbox_y + list(clip_y[::-1]),
        fill="toself",
        mode="none",
        fillcolor="rgba(255,255,255,1)",
        showlegend=False,
    ))

    # border of Switzerland
    border_x, border_y = border_polygon.exterior.xy

    # Add Switzerland border overlay
    fig.add_trace(go.Scatter(
        x=list(border_x),
        y=np.array(border_y),
        mode="lines",
        showlegend=False,
        line=dict(color="black", width=1.5)
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

    width = width or 800
    # Layout adjustments
    fig.update_layout(
        uirevision='true',
        autosize=True,
        height=1 / 1.4 * width + 80,
        modebar_remove=["pan2d", "lasso2d", "resetScale2d", "fullscreen",
                        "zoomIn2d", "zoomOut2d", "toimage"],
        dragmode='select',
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
        margin=dict(l=0, r=0, t=0, b=50)
    )

    return fig


def prob_and_mag_plot(times: pd.Series,
                      probabilities: pd.Series,
                      catalog: pd.DataFrame,
                      current_time: datetime,
                      avg_p: float | None,
                      ylabel: str) -> plt.Figure:

    probabilities = probabilities

    current_prob = probabilities[times[times == current_time].index].values[0]

    fig, [ax0, ax1] = plt.subplots(figsize=(12, 5),
                                   nrows=2,
                                   height_ratios=[1.0, 0.6],
                                   sharex=True)
    plt.subplots_adjust(hspace=0.0)

    ax0.plot(
        times,
        probabilities,
        color='k'
    )

    ax0.set_ylabel(ylabel)

    # make y axis percentages
    ax0.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    xlim = ax0.get_xlim()
    xrange = xlim[1] - xlim[0]
    current_x = mdates.date2num(current_time)
    current_x_offset = current_x - xlim[0]

    if avg_p is not None:
        ylim = ax0.get_ylim()
        yrange = ylim[1] - ylim[0]
        if current_x_offset > (xrange / 2):
            lta_x_offset = xlim[0] + xrange * 0.2
        else:
            lta_x_offset = xlim[1] - xrange * 0.2
        lta_y_offset = avg_p + yrange * 0.03

        ax0.axhline(avg_p, color='k', linestyle='--')
        ax0.text(lta_x_offset,
                 lta_y_offset,
                 'long-term avg: {:.2%}'.format(avg_p),
                 verticalalignment='bottom',
                 horizontalalignment='left',
                 fontsize=10,
                 backgroundcolor='white',
                 color='k')

    label_current = f"current {current_prob:.2%}"

    ylim = ax0.get_ylim()
    yrange = ylim[1] - ylim[0]
    if (current_prob - ylim[0]) < (yrange / 2):
        current_y_offset = ylim[1] - yrange * 0.05
    else:
        current_y_offset = current_prob - yrange * 0.05

    if current_x_offset > (xrange / 2):
        current_x_pos = current_x - xrange * 0.02
    else:
        current_x_pos = current_x + xrange * 0.005

    ax0.axvline(current_time, color='k')
    ax0.text(current_x_pos,
             current_y_offset,
             label_current,
             rotation=90,
             verticalalignment='top',
             horizontalalignment='left',
             fontsize=10,
             color='k')

    dot_sizes = []
    if not catalog.empty:
        dot_sizes = dot_size([*catalog['magnitude'], 7.5],
                             smallest=10,
                             largest=2600,
                             interpolation_power=3)[:-1]

    ax1.scatter(
        catalog['time'],
        catalog['magnitude'],
        color='none', edgecolor='k', marker='o',
        s=dot_sizes,
        linewidth=0.5
    )
    for evt in catalog.itertuples():
        ax1.plot([evt.time, evt.time], [0, evt.magnitude], color='k', lw=0.5)

    ax1.set_ylabel("Magnitude")
    max_mag = catalog['magnitude'].max()
    if np.isnan(max_mag):
        max_mag = 5
    ax1.set_ylim([2 - 0.2, max_mag + 0.5])
    ax1.grid(axis='y', color='k', alpha=0.1)

    ax1.set_xlim([times.min() - timedelta(days=1),
                  times.max() + timedelta(days=1)])
    ax1.axvline(current_time, color='k')

    return fig
