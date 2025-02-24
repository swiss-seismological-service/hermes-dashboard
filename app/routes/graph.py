from datetime import datetime, timedelta

import cmcrameri.cm as cmc
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from cartopy.io import shapereader
from matplotlib import colors
from seismostats.plots.basics import dot_size
from shapely import Polygon, wkt

from app.client import (get_event_counts, get_forecast, get_forecast_cat,
                        get_forecasts, get_observation_cat)


def p_event_all_ch_extrapolate(simulations: pd.DataFrame,
                               n_simulations: int,
                               delta_m_min: float,
                               beta: float) -> float:

    p_event = []

    sizes = simulations.groupby('catalog_id').size()
    p_event = np.sum(
        1 - np.power(1 - np.exp(-beta * delta_m_min), sizes)
    ) / n_simulations

    return p_event


def plot_rel_map(ratio: np.ndarray,
                 scatter_catalog: pd.DataFrame,
                 m_thresh: float,
                 bounding_polygon: Polygon) -> plt.Figure:

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


forecast_mag_threshold = 5.0
n_simulations = 10000  # n_simulations in model settings!!!!
min_mag_stored = 2.5  # m_thresh in model config!!!!

forecasts = get_forecasts(st.session_state['forecastseries']['oid'])
forecasts = [f for f in forecasts if f['status'] == 'COMPLETED']

forecast = st.select_slider(
    "Select a forecast",
    options=forecasts,
    value=forecasts[0],
    format_func=lambda x: x['starttime']
)

forecast = get_forecast(forecast['oid'])

starttime = datetime.fromisoformat(forecast['starttime'])

modelrun = next((r for r in forecast['modelruns']
                 if r['status'] == 'COMPLETED'
                 and r['modelconfig_name']
                 == st.session_state['model_config']['name']), None)

if modelrun is None:
    st.error('No completed modelruns found for '
             f'"{st.session_state["model_config"]["name"]}".')
    st.stop()

observation_cat = get_observation_cat(forecast)

observation_cat = observation_cat[
    (observation_cat['time'] >= starttime - timedelta(days=365))
    & (observation_cat['time'] < starttime)
    & (observation_cat['magnitude'] >= min_mag_stored)]

forecast_cat = get_forecast_cat(modelrun['oid'])

bounding_polygon = wkt.loads(
    st.session_state['forecastseries']['bounding_polygon'])

ratio = get_event_counts(modelrun['oid'], bounding_polygon, n_simulations)

# probability for all of Switzerland
p_event = p_event_all_ch_extrapolate(forecast_cat,
                                     n_simulations,
                                     forecast_mag_threshold - min_mag_stored,
                                     np.log(10))

my_plot2 = plot_rel_map(ratio,
                        observation_cat,
                        forecast_mag_threshold,
                        bounding_polygon)

st.write(
    f'Probability of observing at least one M≥{forecast_mag_threshold} event'
    f' in Switzerland in the next 7 days: {p_event * 100:.2f}%')

# st.pyplot(my_plot)
st.pyplot(my_plot2)
