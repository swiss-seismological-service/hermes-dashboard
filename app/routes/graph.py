from datetime import datetime, timedelta

import numpy as np
import streamlit as st
from shapely import wkt
from streamlit_js_eval import streamlit_js_eval

from app.analysis import p_event_extrapolate, p_event_ratio_grid
from app.client import (get_event_count_grid, get_forecast,
                        get_forecast_seismicityobservation, get_forecasts,
                        get_forecastseries_event_counts)
from app.plots.plots import plot_rel_map_plotly, prob_and_mag_plot
from app.utils import calculate_selection_polygon, get_border_polygon

# Session State Management ***************************************************
if 'p_map' not in st.session_state:
    st.session_state.p_map = None
else:
    st.session_state.p_map = st.session_state.p_map

if 'selection' not in st.session_state:
    st.session_state.selection = None
else:
    st.session_state.selection = st.session_state.selection

st.session_state.width = streamlit_js_eval(
    js_expressions="window.innerWidth", key='SCR')

if 'p_map_selection' in st.session_state:
    # Check the size of the selection and create a polygon from it
    st.session_state.selection = calculate_selection_polygon(
        st.session_state.p_map_selection['selection'])
else:
    st.session_state.selection = None

# General setup from sidebar selection ****************************************
forecast_mag_threshold = 5.0

n_simulations = \
    st.session_state.forecastseries['model_settings']['n_simulations']
min_mag_stored = st.session_state.model_config['model_parameters']['m_thr']

bounding_polygon = wkt.loads(
    st.session_state['forecastseries']['bounding_polygon'])
selection_polygon = st.session_state.selection or bounding_polygon

forecasts = get_forecasts(st.session_state.forecastseries['oid'])
forecasts = [f for f in forecasts if f['status'] == 'COMPLETED']
if len(forecasts) == 0:
    st.error('No completed forecasts found.')
    st.stop()
if len(forecasts) == 1:
    forecasts = forecasts * 2

# Select forecast to display and gather required data.*************************
forecast_slider = st.select_slider(
    'Select a forecast',
    options=forecasts,
    value=forecasts[0],
    format_func=lambda x: x['starttime'],
    key='forecast_slider'
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


count_grid = get_event_count_grid(current_modelrun['oid'],
                                  bounding_polygon,
                                  n_simulations)

ratio = p_event_ratio_grid(count_grid, n_simulations, 0.0001)

border_polygon = get_border_polygon('Switzerland')

st.session_state.p_map = plot_rel_map_plotly(ratio,
                                             observation_catalog,
                                             bounding_polygon,
                                             selection_polygon,
                                             border_polygon,
                                             st.session_state.width)

st.markdown(
    f'''
    #### Probability increase of at least one \
    M≥{forecast_mag_threshold:.1f} event in the next 7 days \
    on the _{current_time.strftime('%Y-%m-%d')}_.
    ''')

if st.session_state.selection is not None:
    # Update the selection
    min_lon, min_lat, max_lon, max_lat = st.session_state.selection.bounds
    st.session_state.p_map.layout.selections = tuple()
    st.session_state.p_map.add_selection(
        x0=min_lon, y0=min_lat, x1=max_lon, y1=max_lat)
    st.info('Double click on the map to clear selection.')


st.plotly_chart(st.session_state.p_map,
                key='p_map_selection',
                on_select='rerun',
                use_container_width=True)

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
elif len(p_event_series) == 1:
    st.info('Only one forecast found for the selected '
            'forecastseries:  \nProbability of at '
            f'least one M≥{forecast_mag_threshold:.1f} event: '
            f'**{p_event_series["p_event"].values[0] * 100:.2f}%**')
else:
    fig2 = prob_and_mag_plot(p_event_series['starttime'],
                             p_event_series['p_event'],
                             full_observation_catalog,
                             current_time,
                             0.2715,
                             forecast_mag_threshold)
    st.pyplot(fig2)
