import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Smart Traffic Dashboard", layout="wide")
st.title("🚦 Smart Traffic Control System")

st.markdown(
    """
This dashboard can **play video**, **run live vehicle detection**, and **show lane counts**.
If your deployed app showed only green boxes, that was the default bounding box color.
Use the controls below to change or disable it.
"""
)

# ---- Controls ----
run_detection = st.sidebar.checkbox("Run YOLO vehicle detection", value=True)
show_boxes = st.sidebar.checkbox("Show vehicle bounding boxes", value=True)
box_color_name = st.sidebar.selectbox("Bounding box color", ["Green", "Red", "Blue", "Yellow"])

BOX_COLORS = {
    "Green": (0, 255, 0),
    "Red": (0, 0, 255),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
}

LANE_REGIONS = [
    np.array([[0, 0], [700, 0], [700, 400], [0, 400]]),
    np.array([[700, 0], [1280, 0], [1280, 400], [700, 400]]),
]
VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike", "motorcycle"}

# Load YOLO once for better performance
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

# ---- Video upload ----
uploaded_file = st.file_uploader("Upload Traffic Video", type=["mp4"])

if uploaded_file is not None:
    st.success("Video uploaded successfully!")

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    model = load_model() if run_detection else None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        lane_counts = [0, 0]

        if run_detection and model is not None:
            results = model(frame, verbose=False)[0]

            for box in results.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                for i, region in enumerate(LANE_REGIONS):
                    if cv2.pointPolygonTest(region, (cx, cy), False) >= 0:
                        lane_counts[i] += 1

                if show_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLORS[box_color_name], 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        BOX_COLORS[box_color_name],
                        2,
                    )

        for i, region in enumerate(LANE_REGIONS):
            cv2.polylines(frame, [region], True, (255, 0, 0), 2)
            x, y = region[0]
            cv2.putText(
                frame,
                f"Lane {i + 1}: {lane_counts[i]}",
                (x + 10, y + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

        stframe.image(frame, channels="BGR")

    cap.release()

# ---- Traffic data chart ----
st.subheader("📊 Traffic Data")

try:
    data = pd.read_csv("traffic_data.csv", header=None)
    data.columns = ["Lane 1", "Lane 2"]
    st.line_chart(data)
except Exception:
    st.warning("No traffic data available yet.")
