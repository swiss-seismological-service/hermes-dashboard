import io
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import cmcrameri.cm as cmc
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from cartopy.io import shapereader
from matplotlib import colors
from seismostats import ForecastCatalog
from seismostats.io.client import FDSNWSEventClient
from seismostats.plots.basics import dot_size

n_simulations = 10000
date = datetime(2025, 2, 15, 12, 0, 0)
forecast_mag_threshold = 5.0
min_mag_stored = 2.5

# get forecast data
oid = "1e7b8b32-5fe5-4c8e-8391-f7e724567071"
response = requests.get(f"http://localhost:8000/v2/modelruns/{oid}/result")
df = pd.read_csv(io.StringIO(response.text))
forecast_cat = ForecastCatalog(df)
forecast_cat['time'] = pd.to_datetime(forecast_cat['time'])
forecast_cat.rename(columns={"realization_id": "catalog_id", }, inplace=True)

# get observation data
client = FDSNWSEventClient(url="http://eida.ethz.ch/fdsnws/event/1/query")
observation_cat = client.get_events(
    start_time=date - timedelta(days=365),
    end_time=date,
    min_magnitude=2.5)

# past probabilities in all CH
longterm_avg_weekly_p = 0.2715
longterm_bg_weekly_cell_p = 0.0001


# spatial probabilities
bg_rates = pd.read_csv("app/static/SUIhaz2015_rates.csv", index_col=0)
bg_rates.query("in_poly", inplace=True)

bg_lats = np.unique(bg_rates.query("in_poly")["latitude"])
bg_lons = np.unique(bg_rates.query("in_poly")["longitude"])
binsize = np.min(np.diff(bg_lats))

# read ch shape polygon
ch_shape = np.load("app/static/ch_shape_buffer.npy")

ll_proj = ccrs.PlateCarree()  # CRS for raw long/lat

# request data for use by geopandas
resolution = "10m"
category = "cultural"
name = "admin_0_countries"
country = "Switzerland"
shpfilename = shapereader.natural_earth(resolution, category, name)
df = geopandas.read_file(shpfilename)
# get geometry of a country

poly = [df.loc[df["ADMIN"] == country]["geometry"].values[0]]


def p_event_all_ch_extrapolate(simulations, n_simulations, delta_m_min, beta):

    p_event = []

    sizes = simulations.groupby("catalog_id").size()
    p_event = np.sum(
        1 - np.power(1 - np.exp(-beta * delta_m_min), sizes)
    ) / n_simulations

    return p_event


def prepare_forecast_map(simulations, n_simulations):
    forecast = simulations.groupby(
        ["latitude_cut", "longitude_cut"]).size().unstack() / n_simulations
    ratio = ((1 - np.exp(-forecast.values)) / longterm_bg_weekly_cell_p)
    ratio = np.clip(ratio, a_min=1, a_max=None)
    return ratio


def plot_rel_map(simulations, m_thresh, cmap,
                 n_simulations=100000, scatter_catalog=pd.DataFrame()):

    if cmap == cmc.batlowK or cmap == cmc.batlow:
        scatter_color = 'w'
    else:
        scatter_color = 'k'

    # make array of latitude bins and longitude bins
    lat_bins = np.arange(bg_lats[0] - binsize / 2,
                         bg_lats[-1] + binsize, binsize)
    lon_bins = np.arange(bg_lons[0] - binsize / 2,
                         bg_lons[-1] + binsize, binsize)

    # use pandas cut to put simulations in same grid as bg_rates
    simulations["latitude_cut"] = pd.cut(simulations.latitude, lat_bins)
    simulations["longitude_cut"] = pd.cut(simulations.longitude, lon_bins)

    # use logarithmic colorscale
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1)

    # use discrete colormap with 5 colors
    norm_min = 0.5
    norm_max = 3.1
    norm = colors.BoundaryNorm(np.arange(norm_min, norm_max, 0.02), cmap.N)
    cmap.set_under(cmap(0.0))

    ############
    # MAP PLOT #
    ############

    ratio = prepare_forecast_map(simulations, n_simulations)
    ax.set_title("Earthquake forecast for Switzerland on {}".format(
        date.strftime('%d. %m. %Y %H:%M')))

    im = ax.imshow(
        np.log10(ratio),
        origin='lower',
        cmap=cmap,
        norm=norm,
        extent=[lon_bins[0], lon_bins[-1], lat_bins[0], lat_bins[-1]],
        interpolation='bilinear',

    )

    dot_sizes = dot_size([*scatter_catalog["magnitude"], 7.5],
                         smallest=10, largest=2600, interpolation_power=3)[:-1]

    ax.scatter(
        scatter_catalog['longitude'],
        scatter_catalog['latitude'],
        color='none', edgecolor=scatter_color, marker='o',
        s=dot_sizes,
        linewidth=0.5,
    )

    # show only the part inside the polygon
    chpoly = plt.Polygon(
        np.flip(ch_shape), edgecolor='k', facecolor='none', lw=0, zorder=10,
        transform=ax.transData, figure=fig
    )
    ax.add_patch(chpoly)
    im.set_clip_path(chpoly)
    ax.plot(
        poly[0].exterior.xy[0],
        poly[0].exterior.xy[1],
        color=scatter_color, lw=1, zorder=10
    )

    cbar = plt.colorbar(
        im, label=f"Probability increase to a normal day\nof at least one M≥{
            m_thresh:.1f} event in 7 days",
        shrink=0.5, orientation='horizontal', pad=0.0, extend='max'
    )
    cbar.ax.minorticks_off()
    cbar_ticks = np.arange(norm_min, norm_max, 0.5)
    cbar_labels = [r"{}$\times$".format(np.format_float_positional(
        10**i, trim='-', precision=0)) for i in cbar_ticks]
    if cbar_ticks[0] > 0:
        cbar_labels[0] = "1-" + cbar_labels[0]
    cbar_labels[-1] = "≥" + cbar_labels[-1]
    cbar.set_ticks(cbar_ticks, labels=cbar_labels)
    ax.set_aspect(1.4)
    ax.axis('off')
    return fig


# probability for all of Switzerland
p_event = p_event_all_ch_extrapolate(
    forecast_cat, n_simulations,
    forecast_mag_threshold - min_mag_stored, np.log(10))
my_plot = plot_rel_map(forecast_cat, 5.0,
                       cmc.lajolla_r,
                       n_simulations=n_simulations,
                       scatter_catalog=observation_cat)


st.write(
    f"Probability of observing at least one M≥{forecast_mag_threshold} event"
    f" in Switzerland in the next 7 days: {p_event * 100:.2f}%")
st.pyplot(my_plot)
