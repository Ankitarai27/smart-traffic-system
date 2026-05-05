import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np

st.title("🚦 Smart Traffic System")

uploaded_file = st.file_uploader("Upload Traffic Video", type=["mp4"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    model = YOLO("yolov8n.pt")

    LANE_REGIONS = [
        np.array([[0, 0], [700, 0], [700, 400], [0, 400]]),
        np.array([[700, 0], [1280, 0], [1280, 400], [700, 400]])
    ]

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        results = model(frame)[0]

        lane_counts = [0, 0]

        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in ["car", "truck", "bus", "motorbike"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2)//2
                cy = (y1 + y2)//2

                for i, region in enumerate(LANE_REGIONS):
                    if cv2.pointPolygonTest(region, (cx, cy), False) >= 0:
                        lane_counts[i] += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        for i, region in enumerate(LANE_REGIONS):
            cv2.polylines(frame, [region], True, (255,0,0), 2)
            x, y = region[0]
            cv2.putText(frame, f"Lane {i+1}: {lane_counts[i]}",
                        (x+10, y+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()