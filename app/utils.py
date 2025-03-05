import streamlit as st
from shapely.geometry import box


def pprint_snake(text: str) -> str:
    """
    Pretty Print a snake_case string into a Title Case string
    """
    return " ".join([word for word in text.split("_")])


def selection_callback(component: str):
    if component in st.session_state and \
            len(st.session_state[component]['selection']['box']) > 0:
        selected_box = st.session_state[component]['selection']['box'][0]

        # Ensure correct order: (minx, miny, maxx, maxy)
        miny, maxy = sorted(selected_box['x'])
        minx, maxx = sorted(selected_box['y'])

        # Update selection and bounding polygon
        st.session_state.selection = box(minx, miny, maxx, maxy)
    else:
        st.session_state.selection = None


def constrain_selection():
    pass
