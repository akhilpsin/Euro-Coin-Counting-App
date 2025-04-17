import cv2
from ultralytics import YOLO
from collections import defaultdict

# Load your trained YOLO model
model = YOLO("euro_coin_detector/weights/best.pt")

# Coin class names and values
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

# Load image
image_path = r"test_image\20250416_224722.jpg"  # Change this to your image file
image = cv2.imread(image_path)

# Inference
results = model(image)[0]

# Counting logic
counts = defaultdict(int)
total_value = 0.0

for box in results.boxes:
    cls_id = int(box.cls)
    conf = float(box.conf)
    label, value = coin_info[cls_id]
    counts[label] += 1
    total_value += value

    # Draw box + label
    xyxy = box.xyxy[0].cpu().numpy().astype(int)
    cv2.rectangle(image, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 255, 0), 2)
    cv2.putText(image, f"{label} ({conf:.2f})", tuple(xyxy[:2] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Add summary text on image
y_offset = 30
for label, count in counts.items():
    cv2.putText(image, f"{label}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 25
cv2.putText(image, f"Total: €{total_value:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# Show result
cv2.imshow("Coin Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
