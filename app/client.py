
import io
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from seismostats import Catalog, ForecastCatalog
from shapely import Polygon

from app import get_config


@st.cache_data
def get_forecasts(forecastseries_oid: str) -> list:
    response = requests.get(f'{get_config().WEBSERVICE_URL}'
                            '/v1/forecastseries/'
                            f'{forecastseries_oid}'
                            '/forecasts'
                            )

    if not response.ok or response.json() == []:
        st.error('No completed forecasts found for '
                 f'"{st.session_state["forecastseries"]["name"]}".')
        st.stop()

    response = response.json()
    response.sort(key=lambda x: datetime.strptime(
        x['starttime'], '%Y-%m-%dT%H:%M:%S'), reverse=False)
    return response


@st.cache_data
def get_forecast(forecast_oid: str) -> dict:
    response = requests.get(f'{get_config().WEBSERVICE_URL}'
                            '/v1/forecasts/'
                            f'{forecast_oid}'
                            )

    if not response.ok:
        st.error('No forecast found for '
                 f'"{st.session_state["forecast"]["name"]}".')
        st.stop()

    return response.json()


@st.cache_resource
def get_observation_cat(forecast: dict) -> Catalog:
    response = requests.get(f'{get_config().WEBSERVICE_URL}'
                            '/v1/forecasts/'
                            f'{forecast["oid"]}'
                            '/seismicityobservations')
    # get observation data
    return Catalog.from_quakeml(response.text)


@st.cache_resource
def get_forecast_cat(modelrun_oid: str) -> ForecastCatalog:
    # get forecast data
    response = requests.get(
        f'{get_config().WEBSERVICE_URL}/v2/modelruns/{modelrun_oid}/result')
    df = pd.read_csv(io.StringIO(response.text))
    forecast_cat = ForecastCatalog(df)
    forecast_cat['time'] = pd.to_datetime(forecast_cat['time'])
    forecast_cat.rename(
        columns={'realization_id': 'catalog_id', }, inplace=True)
    return forecast_cat


def get_event_counts(modelrun_oid: str, geometry: Polygon, n_simulations):
    min_lat, min_lon, max_lat, max_lon = geometry.bounds
    bin = 0.05  # approximate bin size
    res_lon = (max_lon - min_lon) / (round((max_lon - min_lon) / bin))
    res_lat = (max_lat - min_lat) / (round((max_lat - min_lat) / bin))

    response = requests.get(
        f'{get_config().WEBSERVICE_URL}/v2/modelruns/'
        f'{modelrun_oid}/eventcounts',
        params={
            'min_lon': min_lon,
            'min_lat': min_lat,
            'max_lon': max_lon,
            'max_lat': max_lat,
            'res_lon': res_lon,
            'res_lat': res_lat,
        }
    )
    df = pd.read_csv(io.StringIO(response.text))

    matrix_df = df.pivot(index="grid_lat",
                         columns="grid_lon",
                         values="point_count").fillna(0)

    longterm_bg_weekly_cell_p = 0.0001

    ratio = ((1 - np.exp(-matrix_df.values / n_simulations))
             / longterm_bg_weekly_cell_p)

    return np.clip(ratio, a_min=1, a_max=None)
