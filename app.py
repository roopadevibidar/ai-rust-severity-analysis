"""
Rust Inspection System: AI-assisted corrosion inspection for plant assets.

USAGE:
    streamlit run app.py
Then open the URL it prints (default http://localhost:8501).

Records (assets, inspections, photos, work orders) are stored in data/.
"""

import streamlit as st

from core import db

st.set_page_config(page_title="Rust Inspection System", page_icon="🔩", layout="wide")
db.init()

pages = {
    "Operations": [
        st.Page("views/dashboard.py", title="Dashboard", icon="📊", default=True),
        st.Page("views/new_inspection.py", title="New inspection", icon="📷"),
        st.Page("views/work_orders.py", title="Work orders", icon="🛠️"),
    ],
    "Equipment": [st.Page("views/assets.py", title="Assets", icon="🏭")],
    "Tools": [
        st.Page("views/quick_check.py", title="Quick check", icon="🔍"),
        st.Page("views/settings.py", title="Settings", icon="⚙️"),
    ],
}
page = st.navigation(pages)

settings = db.get_settings()
st.sidebar.caption(" · ".join(p for p in (settings["company"], settings["plant_name"]) if p))
page.run()
