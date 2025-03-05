from datetime import datetime, timedelta
from functools import partial
from re import S

import numpy as np
import streamlit as st
from plotly.graph_objs.layout import Selection
from shapely import box, wkt

from app.analysis import p_event_extrapolate, p_event_ratio_grid
from app.client import (get_event_count_grid, get_forecast, get_forecast_cat,
                        get_forecast_seismicityobservation, get_forecasts,
                        get_forecastseries_event_counts)
from app.plots.plots import plot_rel_map_plotly, prob_and_mag_plot
from app.utils import selection_callback

# General setup from sidebar selection ****************************************
if 'selection' not in st.session_state:
    st.session_state.selection = None

if 'plotly_map' not in st.session_state:
    st.session_state.plotly_map = None

if st.session_state.plotly_map is not None and \
        st.session_state.selection is not None:
    min_lat, min_lon, max_lat, max_lon = st.session_state.selection.bounds

    # Compute current width, height, and area
    width = max_lon - min_lon
    height = max_lat - min_lat
    area = width * height

    # If the area is already ≥ 1, return as is
    if area < 1:
        # Scaling factor to make the area exactly 1 while maintaining aspect ratio
        scale_factor = (1 / area) ** 0.5  # Square root to maintain proportions

        # Compute new width and height
        new_width = width * scale_factor
        new_height = height * scale_factor

        # Keep min_lat, min_lon fixed and adjust max_lat, max_lon
        new_max_lat = min_lat + new_height
        new_max_lon = min_lon + new_width

        st.session_state.plotly_map.layout.selections = tuple()
        st.session_state.plotly_map.add_selection(
            x0=min_lon, y0=min_lat, x1=new_max_lon, y1=new_max_lat)

        # ... continue from here

forecast_mag_threshold = 5.0
n_simulations = \
    st.session_state.forecastseries['model_settings']['n_simulations']
min_mag_stored = st.session_state.model_config['model_parameters']['m_thr']

bounding_polygon = wkt.loads(
    st.session_state['forecastseries']['bounding_polygon'])
selection_polygon = st.session_state.selection or bounding_polygon

forecasts = get_forecasts(st.session_state.forecastseries['oid'])
forecasts = [f for f in forecasts if f['status'] == 'COMPLETED']

# Select forecast to display and gather required data.*************************
forecast_slider = st.select_slider(
    'Select a forecast',
    options=forecasts,
    value=forecasts[0],
    format_func=lambda x: x['starttime']
)

current_forecast = get_forecast(forecast_slider['oid'])
current_time = datetime.fromisoformat(current_forecast['starttime'])
current_modelrun = next((mr for mr in current_forecast['modelruns']
                         if mr['status'] == 'COMPLETED'
                         and mr['modelconfig_name']
                         == st.session_state.model_config['name']), None)

if current_modelrun is None:
    st.error('No completed modelruns found for '
             f'"{st.session_state.model_config["name"]}".')
    st.stop()


# Plot the map of the current forecast. ***************************************
observation_catalog = get_forecast_seismicityobservation(
    current_forecast['oid'],
    current_time - timedelta(days=365),
    current_time,
    bounding_polygon,
    min_mag=min_mag_stored)

forecast_catalog = get_forecast_cat(current_modelrun['oid'])


count_grid = get_event_count_grid(current_modelrun['oid'],
                                  bounding_polygon,
                                  n_simulations)

ratio = p_event_ratio_grid(count_grid, n_simulations, 0.0001)

if st.session_state['plotly_map'] is None:
    st.session_state['plotly_map'] = \
        plot_rel_map_plotly(ratio,
                            observation_catalog,
                            forecast_mag_threshold,
                            bounding_polygon,
                            selection_polygon)

st.plotly_chart(st.session_state.plotly_map,
                key='mymap',
                on_select=partial(selection_callback, 'mymap'),
                use_container_width=True,
                )

# Plot the probability timeseries and the event count timeseries. *************
full_observation_catalog = get_forecast_seismicityobservation(
    max(forecasts, key=lambda x: x['starttime'])['oid'],
    min(forecasts, key=lambda x: x['starttime'])['starttime'],
    max(forecasts, key=lambda x: x['starttime'])['starttime'],
    selection_polygon,
    min_mag=2)

eventcounts = get_forecastseries_event_counts(
    st.session_state.forecastseries['oid'],
    st.session_state.model_config['oid'],
    selection_polygon)

p_event_series = p_event_extrapolate(eventcounts,
                                     n_simulations,
                                     forecast_mag_threshold - min_mag_stored,
                                     np.log(10))

if p_event_series.empty:
    st.error('No event counts found for the selected forecast or selection.')
    st.stop()
else:
    fig2 = prob_and_mag_plot(p_event_series['starttime'],
                             p_event_series['p_event'],
                             full_observation_catalog,
                             current_time,
                             0.2715,
                             forecast_mag_threshold)
    st.pyplot(fig2)
