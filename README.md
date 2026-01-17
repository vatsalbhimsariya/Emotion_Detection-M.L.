# Emotion Detection using Machine Learning

**Author:** Vatsal Bhimsariya  
**GitHub:** https://github.com/vatsalbhimsariya  
**Live:** https://huggingface.co/spaces/vatb/emotion-detection
---

## 📌 Project Overview

This project is a **facial emotion detection system** that classifies human emotions into **seven categories**:

- Angry
- Disgusted
- Fearful
- Happy
- Neutral
- Sad
- Surprised

The system uses a **Convolutional Neural Network (CNN)** trained on grayscale facial images and performs **real-time emotion detection** using a webcam.

---

## 🧠 Model & Dataset

- The model architecture is based on a deep CNN.
- The model was originally trained on the **FER-2013** dataset.
- The dataset contains **48×48 grayscale images** of faces labeled with emotions.

⚠️ **Note:**  
This repository uses a **pre-trained model (`model.h5`)**.  
The dataset is **not required** to run emotion prediction.

---

## ⚙️ Technologies Used

- Python 3
- TensorFlow / Keras
- OpenCV
- NumPy
- Haar Cascade for face detection

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/vatsalbhimsariya/Emotion_Detection-M.L.
cd Emotion_Detection-M.L.
```
2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
3️⃣ Run Emotion Detection (Webcam)
```bash
cd src
python emotions.py --mode display
```
📌 A native OpenCV window will open showing the webcam feed.
Press q to close the webcam window.

## 🌐 Live Demo (Emotion Detection)

Try the deployed web app here:  
```
👉 https://huggingface.co/spaces/vatb/emotion-detection
```
Upload an image and see the model predict your facial emotion in real time!

