import streamlit as st

st.title("Test Streamlit App")
st.write("Hello, this is a test!")

import pandas as pd
import numpy as np

# Create some test data
data = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=10),
    'value': np.random.randn(10).cumsum()
})

# Display a chart
st.line_chart(data.set_index('date'))

st.write("If you can see this, Streamlit is working properly!") 