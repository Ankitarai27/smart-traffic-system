import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Smart Traffic Dashboard", layout="wide")
st.title("🚦 Smart Traffic Control System (Live Analytics)")

st.markdown("Upload a video to see **live detection + lane counting** while it processes.")

st.caption("If deployment shows old errors, force a redeploy/restart to pick latest commit.")

show_boxes = st.sidebar.checkbox("Show vehicle bounding boxes", value=True)
box_color_name = st.sidebar.selectbox("Bounding box color", ["Green", "Red", "Blue", "Yellow"])
process_every_n = st.sidebar.slider("Process every Nth frame", min_value=1, max_value=8, value=2)
infer_size = st.sidebar.select_slider("Inference resolution", options=[320, 416, 512, 640], value=416)
save_output = st.sidebar.checkbox("Save and show analyzed video after processing", value=False)

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


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


uploaded_file = st.file_uploader("Upload Traffic Video", type=["mp4"])

if uploaded_file is not None:
    st.success("Video uploaded successfully!")
    video_bytes = uploaded_file.getvalue()

    source_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    source_file.write(video_bytes)
    source_path = source_file.name
    source_file.close()

    cap = cv2.VideoCapture(source_path)
    model = load_model()

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = 1280, 720

    writer = None
    output_path = None
    if save_output:
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_path = output_file.name
        output_file.close()
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    live_frame = st.empty()
    progress = st.progress(0, text="Running live analytics...")

    frame_index = 0
    cached_lane_counts = [0, 0]
    last_boxes = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))

        if frame_index % process_every_n == 0:
            lane_counts = [0, 0]
            detected_boxes = []
            results = model.predict(frame, imgsz=infer_size, verbose=False)[0]

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

                detected_boxes.append((x1, y1, x2, y2, label))

            cached_lane_counts = lane_counts
            last_boxes = detected_boxes

        if show_boxes:
            for x1, y1, x2, y2, label in last_boxes:
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
            cv2.putText(frame, f"Lane {i + 1}: {cached_lane_counts[i]}", (x + 10, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        live_frame.image(frame, channels="BGR", caption="Live Analytics")
        time.sleep(0.01)

        if writer is not None:
            writer.write(frame)

        frame_index += 1
        if frame_count > 0:
            progress.progress(min(frame_index / frame_count, 1.0), text="Running live analytics...")

    cap.release()
    if writer is not None:
        writer.release()
    progress.empty()

    if save_output and output_path:
        st.success("Analyzed video ready below.")
        with open(output_path, "rb") as f:
            analyzed_bytes = f.read()
        st.video(analyzed_bytes)

    for tmp_path in [source_path, output_path]:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass

st.subheader("📊 Traffic Data")

try:
    data = pd.read_csv("traffic_data.csv", header=None)
    data.columns = ["Lane 1", "Lane 2"]
    st.line_chart(data)
except Exception:
    st.warning("No traffic data available yet.")
