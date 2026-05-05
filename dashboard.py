import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Smart Traffic Dashboard", layout="wide")
st.title("🚦 Smart Traffic Control System")

st.markdown("Upload a video to see **live detection + lane counting + emergency priority**.")

# Sidebar
show_boxes = st.sidebar.checkbox("Show bounding boxes", True)
process_every_n = st.sidebar.slider("Process every N frames", 1, 8, 3)
infer_size = st.sidebar.select_slider("Resolution", [320, 416, 512, 640], value=416)
save_output = st.sidebar.checkbox("Save analyzed video", True)

BOX_COLOR = (0, 255, 0)

LANE_REGIONS = [
    np.array([[0, 0], [700, 0], [700, 400], [0, 400]]),
    np.array([[700, 0], [1280, 0], [1280, 400], [700, 400]])
]

VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike", "motorcycle"}


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


uploaded_file = st.file_uploader("Upload Traffic Video", type=["mp4"])

if uploaded_file:

    st.success("Video uploaded!")

    # Save uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    model = load_model()

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1 or fps > 60:
        fps = 25

    width, height = 1280, 720
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video
writer = None
output_path = None

if save_output:
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = output_file.name

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1 or fps > 60:
        fps = 25

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        st.error("❌ VideoWriter failed to open")
    stframe = st.empty()
    progress = st.progress(0)

    frame_index = 0
    cached_lane_counts = [0, 0]
    last_boxes = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))

        # Run detection every N frames
        if frame_index % process_every_n == 0:
            lane_counts = [0, 0]
            detected_boxes = []

            results = model.predict(
                frame,
                imgsz=infer_size,
                conf=0.4,
                verbose=False
            )[0]

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

        # Default signal logic
        active_lane = int(np.argmax(cached_lane_counts))

        # 🚑 Emergency detection (bus = ambulance simulation)
        emergency_detected = any(label == "bus" for _, _, _, _, label in last_boxes)

        if emergency_detected:
            active_lane = int(np.argmax(cached_lane_counts))
            cv2.putText(frame, "🚑 EMERGENCY VEHICLE!",
                        (350, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 3)

        # Draw boxes
        if show_boxes:
            for x1, y1, x2, y2, label in last_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)

        # Draw lanes + signal
        for i, region in enumerate(LANE_REGIONS):
            color = (0, 255, 0) if i == active_lane else (0, 0, 255)

            cv2.polylines(frame, [region], True, (255, 0, 0), 2)
            x, y = region[0]

            cv2.putText(frame, f"Lane {i+1}: {cached_lane_counts[i]}",
                        (x + 10, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.putText(frame,
                        "GREEN" if i == active_lane else "RED",
                        (x + 10, y + 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        3)

        # Save traffic data
        with open("traffic_data.csv", "a") as f:
            f.write(f"{cached_lane_counts[0]},{cached_lane_counts[1]}\n")

        # Show frame
        stframe.image(frame, channels="BGR")

        if writer:
            writer.write(frame)

        frame_index += 1

        if frame_count > 0:
            progress.progress(min(frame_index / frame_count, 1.0))

    cap.release()

    if writer:
        writer.release()

    progress.empty()

    # ✅ FIXED VIDEO DISPLAY
    if save_output and output_path:
        st.success("Analyzed video ready")

        time.sleep(2)

        file_size = Path(output_path).stat().st_size
        st.write(f"Output video size: {file_size} bytes")

        with open(output_path, "rb") as f:
            video_bytes = f.read()
            st.video(video_bytes, format="video/mp4")
    # Cleanup
    try:
        Path(video_path).unlink()
    except:
        pass


# 📊 Graph
st.subheader("📊 Traffic Data")

try:
    data = pd.read_csv("traffic_data.csv", header=None)
    data.columns = ["Lane 1", "Lane 2"]
    st.line_chart(data)
except:
    st.warning("No traffic data yet")