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
Use **Smooth playback** for a normal running video.
Use **Analytics mode** for lane counting and detection overlays.
For faster analytics, increase frame skipping and reduce inference resolution.
"""
)

playback_mode = st.sidebar.radio("Playback mode", ["Smooth playback", "Analytics mode"], index=0)
run_detection = st.sidebar.checkbox("Run YOLO vehicle detection", value=True)
show_boxes = st.sidebar.checkbox("Show vehicle bounding boxes", value=True)
box_color_name = st.sidebar.selectbox("Bounding box color", ["Green", "Red", "Blue", "Yellow"])
process_every_n = st.sidebar.slider("Process every Nth frame", min_value=1, max_value=12, value=4)
display_every_n = st.sidebar.slider("Display every Nth frame", min_value=1, max_value=6, value=2)
infer_size = st.sidebar.select_slider("Inference resolution", options=[320, 416, 512, 640], value=416)

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

    if playback_mode == "Smooth playback":
        st.video(video_bytes)

    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_bytes)

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        stats_box = st.sidebar.empty()

        model = load_model() if run_detection else None

        frame_index = 0
        cached_lane_counts = [0, 0]
        processed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))
            should_process = frame_index % process_every_n == 0
            should_display = frame_index % display_every_n == 0

            if should_process:
                processed_frames += 1
                lane_counts = [0, 0]

                if run_detection and model is not None:
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

                        if show_boxes and should_display:
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

                cached_lane_counts = lane_counts

            if should_display:
                for i, region in enumerate(LANE_REGIONS):
                    cv2.polylines(frame, [region], True, (255, 0, 0), 2)
                    x, y = region[0]
                    cv2.putText(
                        frame,
                        f"Lane {i + 1}: {cached_lane_counts[i]}",
                        (x + 10, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2,
                    )

                stframe.image(frame, channels="BGR")

            frame_index += 1

            if frame_index % 30 == 0:
                stats_box.info(
                    f"Frames read: {frame_index} | Frames processed: {processed_frames} | "
                    f"Process every: {process_every_n} | Display every: {display_every_n}"
                )

        cap.release()

st.subheader("📊 Traffic Data")

try:
    data = pd.read_csv("traffic_data.csv", header=None)
    data.columns = ["Lane 1", "Lane 2"]
    st.line_chart(data)
except Exception:
    st.warning("No traffic data available yet.")
