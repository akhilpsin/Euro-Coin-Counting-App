import cv2
from ultralytics import YOLO
from collections import defaultdict

# Load trained YOLOv8 model
model = YOLO("C:/Users/akhil/runs/detect/euro_coin_detector4/weights/best.pt")

# Map class IDs to coin labels and Euro values
coin_info = {
    0: ("1 cent", 0.01),
    1: ("2 cent", 0.02),
    2: ("5 cent", 0.05),
    3: ("10 cent", 0.10),
    4: ("20 cent", 0.20),
    5: ("50 cent", 0.50),
    6: ("1 euro", 1.00),
    7: ("2 euro", 2.00),
}

# Open webcam (0 for default cam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)[0]

    # Count coins
    counts = defaultdict(int)
    total_value = 0.0

    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        label, value = coin_info[cls_id]
        counts[label] += 1
        total_value += value

        # Draw box and label
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", tuple(xyxy[:2] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display total count and value
    y_offset = 30
    for label, count in counts.items():
        cv2.putText(frame, f"{label}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    cv2.putText(frame, f"Total: €{total_value:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Euro Coin Counter", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
