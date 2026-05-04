import streamlit as st
import pandas as pd
import os

st.title("🚦 Smart Traffic Dashboard")

# ✅ Check if file exists
if not os.path.exists("traffic_data.csv"):
    st.warning("No data yet. Run main.py first 🚗")
else:
    data = pd.read_csv("traffic_data.csv", header=None)
    data.columns = ["Lane 1", "Lane 2"]

    st.line_chart(data)

    st.write("Latest Data:")
    st.write(data.tail())