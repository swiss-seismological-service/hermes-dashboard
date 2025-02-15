import streamlit as st

st.set_page_config(
    page_icon="app/static/favicon.png",
)

home = st.Page(page="routes/home.py",
               title="Dashboard",
               default=True)

subpage = st.Page(page="routes/sub.py",
                  title="Subpage")

graph = st.Page(page="routes/graph.py",
                title="Graph")


pg = st.navigation(
    {
        "Menu": [home, subpage, graph],
        "": []
    }
)

pg.run()
