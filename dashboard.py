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

# --- SIDEBAR ---
show_boxes = st.sidebar.checkbox("Show boxes", True)

# --- CONSTANTS ---
LANE_REGIONS = [
    np.array([[0, 200], [640, 200], [640, 720], [0, 720]]),  # Adjusted for better visibility
    np.array([[640, 200], [1280, 200], [1280, 720], [640, 720]]),
]
VEHICLES = {"car", "truck", "bus", "motorbike", "motorcycle"}

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Save the upload to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close() # Close so OpenCV can open it

    cap = cv2.VideoCapture(tfile.name)
    
    # Video Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # We use 'avc1' (H.264) which is the most browser-compatible codec
    # If this fails on your machine, it means you don't have the H.264 plugin for OpenCV
    output_path = "final_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'H264') 
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # UI Elements
    st.subheader("🎥 Live AI Processing")
    st_frame = st.empty()
    progress_bar = st.progress(0)
    
    col1, col2, col3 = st.columns(3)
    m1 = col1.empty()
    m2 = col2.empty()
    m3 = col3.empty()

    model = load_model()
    traffic_history = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        lane_counts = [0, 0]

        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label not in VEHICLES: continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            for i, region in enumerate(LANE_REGIONS):
                if cv2.pointPolygonTest(region, (cx, cy), False) >= 0:
                    lane_counts[i] += 1
            
            if show_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Drawing Lanes
        active_lane = int(np.argmax(lane_counts))
        for i, region in enumerate(LANE_REGIONS):
            color = (0, 255, 0) if i == active_lane else (0, 0, 255)
            cv2.polylines(frame, [region], True, (255, 0, 0), 2)
            cv2.putText(frame, f"L{i+1}: {lane_counts[i]}", (region[0][0]+10, region[0][1]+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Update Metrics
        m1.metric("Lane 1", lane_counts[0])
        m2.metric("Lane 2", lane_counts[1])
        m3.metric("Green Signal", f"Lane {active_lane + 1}")
        
        traffic_history.append(lane_counts)

        # 1. SHOW LIVE PREVIEW (Convert BGR to RGB)
        st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
        # 2. WRITE TO FILE
        writer.write(frame)

    cap.release()
    writer.release()
    
    # --- DISPLAY FINAL VIDEO ---
    st.success("✅ Processing Complete!")
    if os.path.exists(output_path):
        with open(output_path, 'rb') as f:
            video_bytes = f.read()
            st.video(video_bytes)
    
    # Save CSV
    df = pd.DataFrame(traffic_history, columns=["Lane 1", "Lane 2"])
    df.to_csv("traffic_data.csv", index=False)
    
    st.subheader("📊 Traffic Trends")
    st.line_chart(df)

    # Cleanup temp file
    os.remove(tfile.name)