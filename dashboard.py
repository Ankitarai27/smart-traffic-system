import streamlit as st
import pandas as pd
import cv2
import tempfile

st.title("🚦 Smart Traffic Control System")

# 📤 Upload video
uploaded_file = st.file_uploader("Upload Traffic Video", type=["mp4"])

if uploaded_file is not None:
    st.success("Video uploaded successfully!")

    # Save video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for display
        frame = cv2.resize(frame, (640, 360))

        # Show video
        stframe.image(frame, channels="BGR")

    cap.release()

# 📊 Show traffic data if available
st.subheader("📊 Traffic Data")

try:
    data = pd.read_csv("traffic_data.csv", header=None)
    data.columns = ["Lane 1", "Lane 2"]
    st.line_chart(data)
except:
    st.warning("No traffic data available yet.")