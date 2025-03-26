import streamlit as st
import os
from datetime import datetime

# Create required directories first
for directory in ['data', 'data/logs', 'data/historical', 'data/models']:
    os.makedirs(directory, exist_ok=True)

# Then configure the app
st.set_page_config(
    page_title="Trading Bot Test",
    page_icon="📈",
    layout="wide"
)

st.title("Trading Bot Dashboard")
st.markdown("## Test Page")
st.success("If you can see this, the app is working correctly!")

# Show current time
st.info(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Show that the directories were created
st.write("Created directories:")
for directory in ['data', 'data/logs', 'data/historical', 'data/models']:
    if os.path.exists(directory):
        st.write(f"✅ {directory}")
    else:
        st.write(f"❌ {directory}") 