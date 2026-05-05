import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Smart Traffic Dashboard", layout="wide")
st.title("🚦 Smart Traffic Live AI")
st.markdown("Upload video and detection output will start immediately.")

# Sidebar
show_boxes = st.sidebar.checkbox("Show boxes", True)

LANE_REGIONS = [
    np.array([[0, 0], [700, 0], [700, 400], [0, 400]]),
    np.array([[700, 0], [1280, 0], [1280, 400], [700, 400]]),
]

VEHICLES = {"car", "truck", "bus", "motorbike", "motorcycle"}


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


uploaded_file = st.file_uploader("Upload Video", type=["mp4"])

if uploaded_file:
    with open("traffic_data.csv", "w") as f:
        f.write("")

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("❌ Video not opening")
        st.stop()

    model = load_model()
    stframe = st.empty()

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    lane1_metric = metric_col1.empty()
    lane2_metric = metric_col2.empty()
    active_metric = metric_col3.empty()

    st.success("▶️ Processing started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))

        results = model(frame, verbose=False)[0]

        lane_counts = [0, 0]
        boxes = []

        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label not in VEHICLES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            for i, region in enumerate(LANE_REGIONS):
                if cv2.pointPolygonTest(region, (cx, cy), False) >= 0:
                    lane_counts[i] += 1

            boxes.append((x1, y1, x2, y2, label))

        if any(label == "bus" for *_, label in boxes):
            cv2.putText(
                frame,
                "🚑 EMERGENCY!",
                (400, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )

        if show_boxes:
            for x1, y1, x2, y2, label in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        active_lane = int(np.argmax(lane_counts))
        for i, region in enumerate(LANE_REGIONS):
            color = (0, 255, 0) if i == active_lane else (0, 0, 255)
            cv2.polylines(frame, [region], True, (255, 0, 0), 2)

            x, y = region[0]
            cv2.putText(
                frame,
                f"Lane {i+1}: {lane_counts[i]}",
                (x + 10, y + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

            cv2.putText(
                frame,
                "GREEN" if i == active_lane else "RED",
                (x + 10, y + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3,
            )

        with open("traffic_data.csv", "a") as f:
            f.write(f"{lane_counts[0]},{lane_counts[1]}\n")

        lane1_metric.metric("Lane 1", lane_counts[0])
        lane2_metric.metric("Lane 2", lane_counts[1])
        active_metric.metric("Active", active_lane + 1)

        stframe.image(frame, channels="BGR")
        time.sleep(0.03)

    cap.release()
    st.success("✅ Done")

st.subheader("📊 Traffic Data")
try:
    data = pd.read_csv("traffic_data.csv", header=None)
    data.columns = ["Lane 1", "Lane 2"]
    st.line_chart(data)
except Exception:
    st.warning("No traffic data yet")
