# 🧮 Euro Coin Counting App

A simple yet powerful **Euro coin counting app** built with **YOLOv8** and **Streamlit**. Upload an image of Euro coins, and the app will automatically detect, count, and calculate the total monetary value based on the coins present in the image.

---

## 🎯 Key Features

- 🔎 **Coin Detection**: Uses a custom-trained YOLOv8 model to identify coins.
- 🧮 **Automatic Counting**: Detects and counts each coin type in the image.
- 💰 **Total Value Calculation**: Instantly displays the total Euro value.
- 📷 **Visual Feedback**: Shows bounding boxes and confidence scores.
- 🚀 **Streamlit UI**: Fast and user-friendly web interface.

---

## 💶 Supported Euro Coins

| Coin Type | Value (€) |
| --------- | --------- |
| 1 cent    | 0.01      |
| 2 cent    | 0.02      |
| 5 cent    | 0.05      |
| 10 cent   | 0.10      |
| 20 cent   | 0.20      |
| 50 cent   | 0.50      |
| 1 euro    | 1.00      |
| 2 euro    | 2.00      |

---

## 🚀 Getting Started

### 📦 Requirements

- Python 3.8+
- pip

### 🛠 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/euro-coin-counter.git
cd euro-coin-counter
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure your YOLOv8 model weights are in:

```
euro_coin_detector/weights/best.pt
```

---

## 🖥️ Running the App

```bash
streamlit run euro_coin_detector_app.py
```

Visit `http://localhost:8501` in your browser.

---

## 🧠 How It Works

- The uploaded image is preprocessed (blur + contrast adjustment).
- The YOLOv8 model detects all visible Euro coins.
- The app counts each coin type and calculates the total value.
- Detected coins are visualized with bounding boxes and labels.

---

## 📂 Project Structure

```
euro-coin-counter/
│
├── euro_coin_detector_app.py         # Streamlit app
├── requirements.txt                  # Dependency file
├── euro_coin_detector/
│   └── weights/
│       └── best.pt                   # YOLOv8 trained model
└── README.md
```

---

## 🧪 Example Output

![example](example_output.png)

---

## 📜 License

This project is licensed under the **MIT License** – feel free to use or modify it!

---

## 🙋‍♂️ Contact

Feel free to open an issue or reach out for questions, feedback, or collaboration ideas.
