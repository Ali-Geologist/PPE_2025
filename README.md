# 🛡️ PPE Detection & Smart Access Control System

This project is an AI-powered safety system that performs real-time **PPE (Personal Protective Equipment) detection**, **face recognition**, and **Arduino-controlled access** using a stepper motor, LEDs, and a webcam. It's designed to ensure only authorized individuals wearing the required safety gear (helmet, mask, and vest) can enter a secure area — such as a lab, construction site, or industrial facility.

---

## 🎯 Key Features

- ✅ Real-time person and PPE detection using a **YOLOv8 model**
- 🎥 Live webcam feed with bounding boxes and confidence scores
- 🧠 Face recognition for logging personnel identity
- 🎙️ Voice feedback using `pyttsx3`
- 🔄 Arduino integration with:
  - Green/Red LED indicators
  - Stepper motor-controlled door/gate
- 📝 Excel log export of recognized persons and timestamps
- 🚫 Automatic denial if multiple people are detected simultaneously

---

## 🧰 Hardware Requirements

- Arduino Uno (or compatible)
- 4-Wire Stepper Motor
- Green & Red LEDs
- Webcam
- USB cable (for Arduino)
- Breadboard & jumper wires
- Compatible PPE (Helmet, Mask, Vest) for detection model

---

## 🛠️ Software & Libraries

- Python 3.8+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV (`opencv-python`)
- `cvzone`
- `pyfirmata`
- `pyttsx3`
- `pandas`, `openpyxl`
- Custom YOLO model: `ppe.pt`
- Face Recognition Utility: `SimpleFacerec`

---

## 🗂️ Project Structure

```
PPE_Detection_Project/
├── PPEDetection-arduino.py         # Main Python script
├── Images/                         # Folder for known face images
├── ppe.pt                          # Trained YOLOv8 model for PPE
├── recognition_log.xlsx            # Auto-generated log file
├── simple_facerec.py               # Face recognition utility
└── README.md
```

---

## ▶️ Running the Project

### 1. Install dependencies

```bash
pip install ultralytics opencv-python cvzone pyfirmata pyttsx3 pandas openpyxl
```

### 2. Upload StandardFirmata to Arduino

Open Arduino IDE → Tools → Board: Arduino Uno → Upload `StandardFirmata` from Examples.

### 3. Run the main script

```bash
python PPEDetection-arduino.py
```

> The system will prompt the first user to enter. Only one person is allowed at a time.

---

## ⚙️ How It Works

1. Webcam captures live feed.
2. YOLO model detects:
   - `Hardhat`
   - `Mask`
   - `Safety Vest`
3. Face is recognized using `SimpleFacerec`.
4. If only one person is present and all PPE are worn:
   - 🔊 "Access granted"
   - ✅ Green LED lights up
   - 🔁 Stepper motor rotates to open the door
5. Otherwise:
   - ❌ Red LED is lit
   - 🔊 "Only one person allowed" or PPE warning

---

## 📦 Output Example

- Real-time display with annotated detections
- Audio feedback
- Logged entries:
```
| Name       | Time                |
|------------|---------------------|
| John Doe   | 2025-06-01 09:31:15 |
```

---

## 📌 Customization

- Add authorized face images in `Images/` folder.
- Modify detection classes or confidence threshold in code.
- Replace `ppe.pt` with an updated or more specialized model.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙋 Author

Developed by Rayan Ali Tlais  
GitHub: ryn2004t(https://github.com/ryn2004t)  
Email: [tlsryn2@gmail.com]

---

## ❤️ Contributions

Pull requests, improvements, and suggestions are welcome!
