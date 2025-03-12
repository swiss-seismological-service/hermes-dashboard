import geopandas
import streamlit as st
from cartopy.io import shapereader
from shapely import Polygon, box


def pprint_snake(text: str) -> str:
    """
    Pretty Print a snake_case string into a Title Case string
    """
    return " ".join([word for word in text.split("_")])


def calculate_selection_polygon(selection: dict,
                                min_size: float = 1) -> Polygon:
    if len(selection['box']) > 0:
        selected_box = selection['box'][0]

        # Ensure correct order: (minx, miny, maxx, maxy)
        min_lon, max_lon = sorted(selected_box['x'])
        min_lat, max_lat = sorted(selected_box['y'])

        # Compute current width, height, and area
        width = max_lon - min_lon
        height = max_lat - min_lat
        area = width * height

        # # If the area is already ≥ 1, return as is
        if area < min_size:
            # Scaling factor, sqrt to maintain proportions
            scale_factor = (min_size / area) ** 0.5

            # Compute new width and height
            add_width = width * scale_factor - (max_lon - min_lon)
            add_height = height * scale_factor - (max_lat - min_lat)

            # Keep min_lat, min_lon fixed and adjust max_lat, max_lon
            max_lat = max_lat + (add_height / 2)
            min_lat = min_lat - (add_height / 2)
            max_lon = max_lon + (add_width / 2)
            min_lon = min_lon - (add_width / 2)

        return box(min_lon, min_lat, max_lon, max_lat)
    return None


@st.cache_data
def get_border_polygon(name: str):
    shpfilename = shapereader.natural_earth('10m',
                                            'cultural',
                                            'admin_0_countries')
    df = geopandas.read_file(shpfilename)
    return df.loc[df['ADMIN'] == name]['geometry'].values[0]


def hash_polygon(polygon: Polygon) -> str:
    return polygon.wkt
