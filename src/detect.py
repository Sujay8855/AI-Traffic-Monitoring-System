import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Import LLM function
from llm import generate_incident_report

# Load YOLO model
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Vehicle classes from COCO dataset
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
vehicle_classes = [2, 3, 5, 7]

# Open video
video_path = "data/traffic.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()

# Counting line position
line_y = 300

# Vehicle counter
vehicle_count = 0

# Store counted IDs
counted_ids = set()

print("Starting AI Traffic Monitoring System...")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video completed")
        break

    # Resize frame
    frame = cv2.resize(frame, (640, 480))

    # Draw counting line
    cv2.line(
        frame,
        (0, line_y),
        (640, line_y),
        (0, 0, 255),
        3
    )

    # Run YOLO detection on GPU
    results = model(frame, device="cuda")

    detections = []

    # Process detections
    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            # Detect only vehicles
            if cls in vehicle_classes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                confidence = float(box.conf[0])

                # DeepSORT format
                detections.append(
                    (
                        [x1, y1, x2 - x1, y2 - y1],
                        confidence,
                        cls
                    )
                )

    # Update tracker
    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        # Bounding box
        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(int, ltrb)

        # Center point
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Default box color
        box_color = (0, 255, 0)

        # Detect line crossing violation
        if center_y > line_y - 5 and center_y < line_y + 5:

            if track_id not in counted_ids:

                counted_ids.add(track_id)

                vehicle_count += 1

                # Violation color
                box_color = (0, 0, 255)

                # Display violation text
                cv2.putText(
                    frame,
                    "VIOLATION",
                    (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    3
                )

                print(f"\nViolation Detected - Vehicle ID: {track_id}")

                # Generate AI report
                report = generate_incident_report(track_id)

                print("\nAI Incident Report:")
                print(report)

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        # Draw center point
        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (255, 0, 0),
            -1
        )

        # Display ID
        cv2.putText(
            frame,
            f"ID: {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2
        )

    # Display counter
    cv2.putText(
        frame,
        f"Vehicle Count: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        3
    )

    # Show output
    cv2.imshow(
        "AI Traffic Monitoring System",
        frame
    )

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
