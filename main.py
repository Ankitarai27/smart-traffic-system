import time
import cv2
import csv
from traffic_logic import update_signal
from ultralytics import YOLO
from siren_detection import detect_siren

last_switch_time = time.time()
current_lane = 0
green_time = 10
last_siren_check = time.time()
siren_detected = False
def draw_signal(frame, active_lane):
    # Positions for 2 lanes
    positions = [(200, 600), (1000, 600)]

    for i, (x, y) in enumerate(positions):
        # Default colors (red)
        red = (0, 0, 255)
        yellow = (0, 255, 255)
        green = (0, 255, 0)

        # If this is active lane → green ON
        if i == active_lane:
            cv2.circle(frame, (x, y), 20, green, -1)
            cv2.circle(frame, (x, y-40), 20, (50,50,50), -1)
            cv2.circle(frame, (x, y-80), 20, (50,50,50), -1)
        else:
            cv2.circle(frame, (x, y-80), 20, red, -1)
            cv2.circle(frame, (x, y-40), 20, (50,50,50), -1)
            cv2.circle(frame, (x, y), 20, (50,50,50), -1)

# Load model
model = YOLO("yolov8n.pt")

EMERGENCY_CLASSES = ["ambulance"]  # (for future use)

# Open video
cap = cv2.VideoCapture("videos/traffic.mp4")

# Define 2 lanes (you can adjust)
import numpy as np

LANE_REGIONS = [
    np.array([[0, 0], [700, 0], [700, 400], [0, 400]]),   # Lane 1 (left road)
    np.array([[700, 0], [1280, 0], [1280, 400], [700, 400]])  # Lane 2 (right road)
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for consistency
    frame = cv2.resize(frame, (1280, 720))

    results = model(frame, conf=0.5)[0]

    lane_counts = [0 for _ in LANE_REGIONS]
    emergency_detected = False
    emergency_lane = None
    
    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # 🚑 Emergency detection (simulate using bus)
        if label == "bus":
            emergency_detected = True

        if label in ["car", "truck", "bus", "motorbike"]:
            for i, region in enumerate(LANE_REGIONS):
                if cv2.pointPolygonTest(region, (cx, cy), False) >= 0:
                    lane_counts[i] += 1

                    if label == "bus":  # treat bus as ambulance
                        emergency_lane = i

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)     
    with open("traffic_data.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(lane_counts)
    # Draw lane regions
    for i, region in enumerate(LANE_REGIONS):
    # Draw polygon
        cv2.polylines(frame, [region], isClosed=True, color=(255,0,0), thickness=2)

        # Text position (first point of polygon)
        x, y = region[0]

        cv2.putText(frame, f"Lane {i+1}: {lane_counts[i]}",
                    (x + 10, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        # 🚦 Decide signal
    # Check siren every 3 seconds (non-blocking feel)
    if time.time() - last_siren_check > 3:
        siren_detected = detect_siren(duration=0.3)
        last_siren_check = time.time()
    # 🚦 Priority logic
    if siren_detected:
        active_lane = 0  # OR choose most crowded lane
        green_time = 30

        cv2.putText(frame, "🚨 SIREN DETECTED!", (400, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    elif emergency_detected and emergency_lane is not None:
        active_lane = emergency_lane
        green_time = 30

        cv2.putText(frame, "🚑 EMERGENCY VEHICLE!", (400, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    else:
        current_time = time.time()

        if current_time - last_switch_time > green_time:
            current_lane, green_time = update_signal(lane_counts)
            last_switch_time = current_time

        active_lane = current_lane

        draw_signal(frame, active_lane)
        # Show result
        cv2.putText(frame, f"Green Lane: {active_lane + 1}",
                    (50, 650),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

        cv2.putText(frame, f"Time: {green_time}s",
                    (50, 690),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

        cv2.imshow("Lane Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
# import cv2
# from ultralytics import YOLO

# # Load YOLO model
# model = YOLO("yolov8n.pt")

# # Open video file
# cap = cv2.VideoCapture("videos/traffic.mp4")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run detection
#     results = model(frame)[0]

#     for box in results.boxes:
#         cls = int(box.cls[0])
#         label = model.names[cls]

#         # Detect only vehicles
#         if label in ["car", "truck", "bus", "motorbike"]:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])

#             # Draw rectangle
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             # Put label
#             cv2.putText(frame, label, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

#     # Show frame
#     cv2.imshow("Vehicle Detection", frame)

#     # Press ESC to exit
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()