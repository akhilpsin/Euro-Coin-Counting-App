import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from collections import defaultdict
from PIL import Image

# Load YOLO model
model = YOLO("euro_coin_detector/weights/best.pt")

# Coin labels and values
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

st.title("🪙 Euro Coin Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image as RGB
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # === Simple Preprocessing ===
    image_cv = cv2.GaussianBlur(image_cv, (3, 3), 0)  # Light blur to remove noise

    # Contrast enhancement with CLAHE
    lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    image_cv = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Run detection
    results = model(image_cv)[0]

    # Counting
    counts = defaultdict(int)
    total_value = 0.0

    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        if conf < 0.4:
            continue

        label, value = coin_info[cls_id]
        counts[label] += 1
        total_value += value

        # Draw box and label
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(image_cv, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 255, 0), 2)
        cv2.putText(image_cv, f"{label} ({conf:.2f})", tuple(xyxy[:2] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show count summary
    y_offset = 30
    for label, count in counts.items():
        cv2.putText(image_cv, f"{label}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    cv2.putText(image_cv, f"Total: €{total_value:.2f}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Convert back to RGB for Streamlit
    image_result = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    st.image(image_result, caption=f"Total: €{total_value:.2f}", use_container_width=True)

else:
    st.info("Please upload an image or take a photo to start detecting.")
