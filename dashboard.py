# import tempfile
# import time
# import cv2
# import numpy as np
# import pandas as pd
# import streamlit as st
# from ultralytics import YOLO
# import os

# st.set_page_config(page_title="Smart Traffic Dashboard", layout="wide")
# st.title("🚦 Smart Traffic Live AI")

# # --- SIDEBAR SETTINGS ---
# st.sidebar.header("Optimization Settings")
# # Higher frame skip = faster processing but less frequent updates
# frame_skip = st.sidebar.slider("Frame Skip (Process every Nth frame)", 1, 20, 5)
# model_res = st.sidebar.selectbox("Model Resolution", [320, 640], index=0)
# show_boxes = st.sidebar.checkbox("Show boxes", True)

# # --- CONSTANTS ---
# LANE_REGIONS = [
#     np.array([[0, 200], [640, 200], [640, 720], [0, 720]]),
#     np.array([[640, 200], [1280, 200], [1280, 720], [640, 720]]),
# ]
# VEHICLES = {"car", "truck", "bus", "motorbike", "motorcycle"}

# @st.cache_resource
# def load_model():
#     return YOLO("yolov8n.pt")

# uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

# if uploaded_file:
#     tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#     tfile.write(uploaded_file.read())
#     tfile.close() 

#     cap = cv2.VideoCapture(tfile.name)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
#     # Use mp4v for writing to ensure compatibility during fast processing
#     output_path = "final_output.mp4"
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
#     writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     st.subheader("🎥 Live AI Processing")
#     st_frame = st.empty()
    
#     col1, col2, col3 = st.columns(3)
#     m1 = col1.empty()
#     m2 = col2.empty()
#     m3 = col3.empty()

#     model = load_model()
#     traffic_history = []
    
#     frame_idx = 0
#     lane_counts = [0, 0] # Keep counts persistent between skips

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # --- SPEED OPTIMIZATION: FRAME SKIPPING ---
#         if frame_idx % frame_skip == 0:
#             # Resize image for faster inference
#             results = model(frame, verbose=False, imgsz=model_res)[0]
#             lane_counts = [0, 0]

#             for box in results.boxes:
#                 cls = int(box.cls[0])
#                 label = model.names[cls]
#                 if label not in VEHICLES: continue

#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

#                 for i, region in enumerate(LANE_REGIONS):
#                     if cv2.pointPolygonTest(region, (cx, cy), False) >= 0:
#                         lane_counts[i] += 1
                
#                 if show_boxes:
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Drawing UI on every frame (smooth visuals)
#         active_lane = int(np.argmax(lane_counts))
#         for i, region in enumerate(LANE_REGIONS):
#             color = (0, 255, 0) if i == active_lane else (0, 0, 255)
#             cv2.polylines(frame, [region], True, (255, 0, 0), 2)
#             cv2.putText(frame, f"L{i+1}: {lane_counts[i]}", (region[0][0]+10, region[0][1]+30), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#         # Update Metrics
#         m1.metric("Lane 1", lane_counts[0])
#         m2.metric("Lane 2", lane_counts[1])
#         m3.metric("Green Signal", f"Lane {active_lane + 1}")
        
#         traffic_history.append(lane_counts)
#         st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
#         writer.write(frame)
        
#         frame_idx += 1

#     cap.release()
#     writer.release()
    
#     st.success("✅ Processing Complete!")
#     if os.path.exists(output_path):
#         with open(output_path, 'rb') as f:
#             st.video(f.read())
    
#     df = pd.DataFrame(traffic_history, columns=["Lane 1", "Lane 2"])
#     st.subheader("📊 Traffic Trends")
#     st.line_chart(df)
#     os.remove(tfile.name)


import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import os

st.set_page_config(page_title="Smart Traffic Dashboard", layout="wide")
st.title("🚦 High-Performance Traffic AI")

# --- TURBO SETTINGS ---
st.sidebar.header("Turbo Settings")
# Defaulting to 6 for maximum speed; increase if still laggy
frame_skip = st.sidebar.slider("Speed Boost (Skip Frames)", 1, 15, 6)
model_res = st.sidebar.selectbox("Inference Resolution (Lower = Faster)", [320, 640], index=0)
show_boxes = st.sidebar.checkbox("Show detection boxes", False)

LANE_REGIONS = [
    np.array([[0, 200], [640, 200], [640, 720], [0, 720]]),
    np.array([[640, 200], [1280, 200], [1280, 720], [640, 720]]),
]
VEHICLES = {"car", "truck", "bus", "motorbike", "motorcycle"}

@st.cache_resource
def load_model():
    # Use Nano model for speed
    return YOLO("yolov8n.pt")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Save the upload to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close() 

    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Pre-calculate output path
    output_path = "fast_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # UI Layout
    st_frame = st.empty()
    m_cols = st.columns(3)
    m1, m2, m3 = m_cols[0].empty(), m_cols[1].empty(), m_cols[2].empty()

    model = load_model()
    frame_idx = 0
    lane_counts = [0, 0]
    traffic_history = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- OPTIMIZATION 1: FRAME SKIPPING ---
        if frame_idx % frame_skip == 0:
            # OPTIMIZATION 2: LOWER RESOLUTION (imgsz=320)
            results = model.predict(frame, verbose=False, imgsz=model_res, conf=0.4)[0]
            lane_counts = [0, 0]

            for box in results.boxes:
                label = model.names[int(box.cls[0])]
                if label in VEHICLES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    for i, region in enumerate(LANE_REGIONS):
                        if cv2.pointPolygonTest(region, (cx, cy), False) >= 0:
                            lane_counts[i] += 1
                    
                    if show_boxes:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw Permanent UI Overlay
        active = int(np.argmax(lane_counts))
        for i, region in enumerate(LANE_REGIONS):
            cv2.polylines(frame, [region], True, (255, 0, 0), 2)
            color = (0, 255, 0) if i == active else (0, 0, 255)
            cv2.putText(frame, f"Lane {i+1}: {lane_counts[i]}", (region[0][0]+20, region[0][1]+40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        writer.write(frame)
        traffic_history.append(lane_counts)

        # --- OPTIMIZATION 3: UI THROTTLING ---
        # Update preview every few frames to reduce browser overhead
        if frame_idx % (frame_skip * 2) == 0:
            # FIXED: Updated use_column_width to use_container_width
            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                           channels="RGB", 
                           use_container_width=True)
            m1.metric("Lane 1 Count", lane_counts[0])
            m2.metric("Lane 2 Count", lane_counts[1])
            m3.metric("Green Status", f"Lane {active + 1}")

        frame_idx += 1

    cap.release()
    writer.release()
    
    st.success("✅ Fast Processing Finished!")
    
    # Display the final video
    with open(output_path, 'rb') as f:
        st.video(f.read())
    
    # Charts
    df = pd.DataFrame(traffic_history, columns=["Lane 1", "Lane 2"])
    st.line_chart(df)
    
    # Cleanup
    os.remove(tfile.name)