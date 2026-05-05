import tempfile
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import os

st.set_page_config(page_title="Smart Traffic Dashboard", layout="wide")
st.title("🚦 Smart Traffic Live AI")

st.markdown("Upload video to see full processed output with vehicle detection.")

# Sidebar Settings
show_boxes = st.sidebar.checkbox("Show boxes", True)

# Constants
LANE_REGIONS = [
    np.array([[0, 0], [700, 0], [700, 400], [0, 400]]),
    np.array([[700, 0], [1280, 0], [1280, 400], [700, 400]]),
]
VEHICLES = {"car", "truck", "bus", "motorbike", "motorcycle"}

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

preview_container = st.container()
with preview_container:
    st.subheader("🎥 Live Detection Preview")
    stframe = st.empty()

if uploaded_file:
    # Reset CSV
    with open("traffic_data.csv", "w") as f:
        f.write("")

    # Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("❌ Video not opening")
        st.stop()

    model = load_model()

    # Get Video Properties
    width = 1280
    height = 720
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1 or fps > 120:
        fps = 25

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output file handling - Use a static name to avoid tempfile permission issues on Windows
    out_path = "processed_output.mp4"
    # 'avc1' is the H.264 codec which is web-friendly
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    st.success("▶️ Processing started...")
    progress = st.progress(0)

    # Metrics Layout
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    lane1_metric = metric_col1.empty()
    lane2_metric = metric_col2.empty()
    active_metric = metric_col3.empty()

    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (width, height))
        results = model(frame, verbose=False)[0]

        lane_counts = [0, 0]
        boxes = []

        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label not in VEHICLES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            for i, region in enumerate(LANE_REGIONS):
                if cv2.pointPolygonTest(region, (cx, cy), False) >= 0:
                    lane_counts[i] += 1

            boxes.append((x1, y1, x2, y2, label))

        # Emergency Detection Logic
        if any(label == "bus" for *_, label in boxes):
            cv2.putText(frame, "🚨 EMERGENCY!", (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        # Drawing
        if show_boxes:
            for x1, y1, x2, y2, label in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        active_lane = int(np.argmax(lane_counts))
        for i, region in enumerate(LANE_REGIONS):
            color = (0, 255, 0) if i == active_lane else (0, 0, 255)
            cv2.polylines(frame, [region], True, (255, 0, 0), 2)
            x, y = region[0]
            cv2.putText(frame, f"Lane {i+1}: {lane_counts[i]}", (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "GREEN" if i == active_lane else "RED", (x + 10, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Update CSV and Metrics
        with open("traffic_data.csv", "a") as f:
            f.write(f"{lane_counts[0]},{lane_counts[1]}\n")

        lane1_metric.metric("Lane 1 Count", lane_counts[0])
        lane2_metric.metric("Lane 2 Count", lane_counts[1])
        active_metric.metric("Active Green Lane", active_lane + 1)

        # Write to File
        writer.write(frame)
        
        # Display Preview
        # Streamlit needs RGB; OpenCV uses BGR
        preview_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(preview_frame, channels="RGB", use_column_width=True)

        frame_id += 1
        if total_frames > 0:
            progress.progress(min(frame_id / total_frames, 1.0))

    cap.release()
    writer.release()

    st.success("✅ Processing complete!")
    
    # Display Result Video
    # Note: If the video still doesn't play in your browser, 
    # it is because of the mp4v codec. 
    # You may need to use 'avc1' or convert with FFmpeg.
    with open(out_path, 'rb') as v_file:
        video_bytes = v_file.read()
        st.video(video_bytes)

# Analytics Section
st.divider()
st.subheader("📊 Historical Traffic Trends")
try:
    df = pd.read_csv("traffic_data.csv", names=["Lane 1", "Lane 2"])
    if not df.empty:
        st.line_chart(df)
    else:
        st.info("Waiting for traffic data...")
except Exception:
    st.warning("No data found to plot.")