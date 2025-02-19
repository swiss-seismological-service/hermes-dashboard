import requests
import streamlit as st

from app import get_config
from app.utils import pprint_snake

st.set_page_config(
    page_icon='app/static/favicon.png',
)

if 'project' not in st.session_state:
    response = requests.get(
        f'http://localhost:8000/v1/projects/{get_config().PROJECT_ID}')
    project = response.json()
    st.session_state['project'] = project

if 'forecastseries_list' not in st.session_state:
    response = requests.get(
        'http://localhost:8000/v1/projects/'
        f'{get_config().PROJECT_ID}/forecastseries')
    forecastseries = response.json()
    st.session_state['forecastseries_list'] = forecastseries

home = st.Page(page='routes/home.py',
               title='Dashboard',
               default=True)

# subpage = st.Page(page='routes/sub.py',
#                   title='Subpage')

graph = st.Page(page='routes/graph.py',
                title='Graph')

pg = st.navigation([home,
                    # subpage,
                    graph
                    ])


with st.sidebar.container():
    # PROJECT
    st.markdown(
        f'''
        ## Project:
        {pprint_snake(st.session_state['project']['name'])}
        ''')

    # FORECAST SERIES
    fs_names = \
        [pprint_snake(f['name'])
         for f in st.session_state['forecastseries_list']]

    st.markdown('## Forecast Series')
    selectbox_fs = st.selectbox('Select a Forecast Series',
                                fs_names,
                                label_visibility='collapsed'
                                )

    selected_index = fs_names.index(selectbox_fs)
    st.session_state['forecastseries'] = \
        st.session_state['forecastseries_list'][selected_index]

    # MODEL CONFIG
    mc_names = \
        [pprint_snake(f['name'])
         for f in st.session_state['forecastseries']['modelconfigs']]

    st.markdown('## Model Configs')
    selectbox_mc = st.selectbox('Select a Model Config',
                                mc_names,
                                label_visibility='collapsed'
                                )
    selected_index = mc_names.index(selectbox_mc)
    st.session_state['model_config'] = \
        st.session_state['forecastseries']['modelconfigs'][selected_index]

    st.divider()

pg.run()
