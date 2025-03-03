from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import streamlit as st
from seismostats.plots.basics import dot_size
from shapely import wkt

from app.client import (get_event_counts, get_forecast, get_forecast_cat,
                        get_forecast_seismicityobservation, get_forecasts,
                        get_forecastseries_event_counts)
from app.plots.plots import plot_rel_map, plot_rel_map_plotly


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


forecast_dfs = forecast_dfs.set_index('starttime')
forecast_dfs['point_count'] = forecast_dfs['point_count'] * 100
fig2, axes = plt.subplots(figsize=(12, 5),
                          nrows=2,
                          height_ratios=[1.0, 0.6],
                          sharex=True)
plt.subplots_adjust(hspace=0.0)

ax = axes[0]

ax.plot(
    forecast_dfs.index,
    forecast_dfs['point_count'],
    color='k'
)

# add text stating today's probability
current_prob = forecast_dfs.loc[starttime, 'point_count']

# # Create a twin axis for the right y-axis
# ax2 = ax.twinx()
# ax2.set_ylim(ax.get_ylim())
# ax2.set_yticks([current_prob])

# # this could not be ideal, I didn't try it for
# # all possible values of forecast_mag_threshold
# if forecast_mag_threshold <= 3.5:
#     ax2.set_yticklabels([f"Now\n{current_prob:.0%}"])
# elif forecast_mag_threshold <= 4.5:
#     ax2.set_yticklabels([f"Now\n{current_prob:.1%}"])
# else:
#     ax2.set_yticklabels([f"Now\n{current_prob:.2%}"])

# longterm_avg_weekly_p = 0.2715
longterm_avg_weekly_p = forecast_dfs['point_count'].mean()

ax.set_ylabel(
    f"Probability of at least one M≥{forecast_mag_threshold:.1f}' \
    'event\nin Switzerland in 7 days")

# make y axis percentages
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

ax.axhline(longterm_avg_weekly_p, color='k', linestyle='--')
ax.text(ax.get_xlim()[0] + 1,
        longterm_avg_weekly_p + 0.001,
        'long-term avg: {:.1%}'.format(longterm_avg_weekly_p),
        verticalalignment='bottom',
        horizontalalignment='left',
        fontsize=10,
        backgroundcolor='white',
        color='k')


if forecast_mag_threshold <= 3.5:
    label_current = f"current {current_prob:.0%}"
elif forecast_mag_threshold <= 4.5:
    label_current = f"current {current_prob:.1%}"
else:
    label_current = f"current {current_prob:.2%}"

ax.axvline(starttime, color='k')
ylim = ax.get_ylim()
xlim = ax.get_xlim()
ax.text(mdates.date2num(starttime) + (xlim[1] - xlim[0]) * 0.01,
        ylim[1] - (ylim[1] - ylim[0]) * 0.05,
        label_current,
        rotation=90,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=10,
        color='k')

dot_sizes = dot_size([*observation_cat['magnitude'], 7.5],
                     smallest=10, largest=2600, interpolation_power=3)[:-1]
ax = axes[1]
ax.scatter(
    observation_cat['time'],
    observation_cat['magnitude'],
    color='none', edgecolor='k', marker='o',
    s=dot_sizes,
    linewidth=0.5,
    label=f"Past M≥{min_mag_stored:.1f} earthquakes"
    # alpha=0.5,
)
for evt in observation_cat.itertuples():
    ax.plot([evt.time, evt.time], [0, evt.magnitude], color='k', lw=0.5)

ax.set_ylabel("Magnitude")
max_mag = observation_cat['magnitude'].max()
ax.set_ylim([min_mag_stored - 0.5, max_mag + 0.5])
ax.grid(axis='y', color='k', alpha=0.1)

for ax in axes[1:]:
    ax.set_xlim([forecast_dfs.index.min() - timedelta(hours=12),
                forecast_dfs.index.max() + timedelta(hours=18)])
    ax.axvline(starttime, color='k')

st.pyplot(fig2)
