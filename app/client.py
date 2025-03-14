
import io
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from seismostats import Catalog
from shapely import Polygon

from app import get_config
from app.utils import hash_polygon


@st.cache_data(ttl=300,
               show_spinner=False)
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


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=3600,
               hash_funcs={Polygon: hash_polygon},
               show_spinner=False)
def get_event_count_grid(modelrun_oid: str,
                         geometry: Polygon):
    min_lon, min_lat, max_lon, max_lat = geometry.bounds
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
                         values="event_count").fillna(0)

    return matrix_df


@st.cache_data(ttl=3600,
               hash_funcs={Polygon: hash_polygon},
               show_spinner=False)
def get_forecast_seismicityobservation(forecast_oid: str,
                                       start_time: datetime,
                                       end_time: datetime,
                                       bounding_polygon: Polygon,
                                       min_mag: float = -10) -> Catalog:

    min_lon, min_lat, max_lon, max_lat = bounding_polygon.bounds

    response = requests.get(
        f'{get_config().WEBSERVICE_URL}/v2/forecasts/'
        f'{forecast_oid}/seismicityobservation',
        params={
            'start_time': start_time,
            'end_time': end_time,
            'min_lon': min_lon,
            'min_lat': min_lat,
            'max_lon': max_lon,
            'max_lat': max_lat,
            'min_mag': min_mag,
        }
    )

    return Catalog.from_quakeml(response.text)


@st.cache_data(ttl=300,
               hash_funcs={Polygon: hash_polygon},
               show_spinner=False)
def get_forecastseries_event_counts(forecastseries_oid: str,
                                    modelconfig_oid: str,
                                    bounding_polygon: Polygon):
    min_lon, min_lat, max_lon, max_lat = bounding_polygon.bounds

    response = requests.get(
        f'{get_config().WEBSERVICE_URL}/v2/forecastseries/'
        f'{forecastseries_oid}/eventcounts',
        params={
            'modelconfig_oid': modelconfig_oid,
            'min_lon': min_lon,
            'min_lat': min_lat,
            'max_lon': max_lon,
            'max_lat': max_lat,
        }
    )

    forecasts = get_forecasts(forecastseries_oid)

    fc_df = pd.DataFrame(forecasts)
    fc_df = fc_df[['oid', 'starttime']]
    fc_df['starttime'] = pd.to_datetime(fc_df['starttime'])

    df = pd.read_csv(io.StringIO(response.text))
    df = df.rename(columns={'forecast_oid': 'oid'})
    df = df.merge(fc_df, on='oid', how='left')

    return df
