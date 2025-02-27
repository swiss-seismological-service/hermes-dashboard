from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from shapely import wkt

from app.client import (get_event_counts, get_forecast, get_forecast_cat,
                        get_forecast_seismicityobservation, get_forecasts,
                        get_forecastseries_event_counts)
from app.plots.plots import create_line_plot, plot_rel_map, plot_rel_map_plotly


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

bounding_polygon = wkt.loads(
    st.session_state['forecastseries']['bounding_polygon'])

observation_cat = get_forecast_seismicityobservation(
    forecast['oid'],
    starttime - timedelta(days=365),
    starttime,
    bounding_polygon,
    min_mag=min_mag_stored)

forecast_cat = get_forecast_cat(modelrun['oid'])


ratio = get_event_counts(modelrun['oid'], bounding_polygon, n_simulations)

# probability for all of Switzerland
# p_event = p_event_all_ch_extrapolate(forecast_cat,
#                                      n_simulations,
#                                      forecast_mag_threshold - min_mag_stored,
#                                      np.log(10))

my_plot = plot_rel_map(ratio,
                       observation_cat,
                       forecast_mag_threshold,
                       bounding_polygon,
                       starttime)

# st.write(
#     f'Probability of observing at least one M≥{forecast_mag_threshold} event'
#     f' in Switzerland in the next 7 days: {p_event * 100:.2f}%')

st.pyplot(my_plot)


plotly_map = plot_rel_map_plotly(ratio,
                                 observation_cat,
                                 forecast_mag_threshold,
                                 bounding_polygon,
                                 starttime)

mymap = st.plotly_chart(plotly_map,
                        key="1",
                        on_select="rerun",
                        use_container_width=True,
                        selection_mode="box")

st.write(mymap.selection)


eventcounts = get_forecastseries_event_counts(
    st.session_state['forecastseries']['oid'],
    st.session_state['model_config']['oid'],
    bounding_polygon)


def p_event(series: pd.Series, n: int, beta, delta_m_min) -> float:
    return np.sum(
        1 - np.power(1 - np.exp(-beta * delta_m_min), series)
    ) / n


def p_event_extrapolate(eventcounts: pd.DataFrame,
                        forecasts: dict,
                        n_simulations: int,
                        delta_m_min: float,
                        beta: float) -> float:

    ps = eventcounts.groupby('forecast_oid',
                             as_index=False)['point_count'].agg(
        lambda x: p_event(x, n_simulations, beta, delta_m_min)
    )

    return ps


ps = p_event_extrapolate(eventcounts,
                         forecasts,
                         n_simulations,
                         forecast_mag_threshold - min_mag_stored,
                         np.log(10))
forecast_dfs = pd.DataFrame(forecasts)[['oid', 'starttime']]

forecast_dfs['starttime'] = pd.to_datetime(forecast_dfs['starttime'])

forecast_dfs = forecast_dfs.merge(
    ps, left_on='oid', right_on='forecast_oid', how='left')
forecast_dfs.drop(columns='forecast_oid', inplace=True)

fig = create_line_plot(forecast_dfs, x_col='starttime', y_col='point_count')
st.pyplot(fig)
