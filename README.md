
# Hand Gesture Recognition Using Leap Motion Dataset

This project aims to build a hand gesture recognition system using the **Leap Motion Gesture Recognition Dataset**. The model classifies hand gestures using a **Convolutional Neural Network (CNN)**, enabling intuitive gesture-based control systems.

## Dataset

The dataset contains images of 10 hand gestures captured using the Leap Motion Controller.

- **Gestures:** 10 different hand gestures (e.g., fist, palm, thumbs-up)
- **Image Size:** 640x240 pixels (grayscale)
- **Total Images:** ~20,000 images

You can download the dataset from [Kaggle: Leap Motion Gesture Recognition Dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog).

## Requirements

Install the necessary libraries:

```bash
pip install numpy pandas tensorflow scikit-learn opencv-python matplotlib
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```

2. **Download and extract the dataset** to the project folder.

3. **Install dependencies** using `pip`.

## Usage

### Training the Model:
To train the model, run:
```bash
python train_model.py
```

### Real-Time Gesture Recognition:
For real-time gesture recognition, run:
```bash
python live_gesture_recognition.py
```

## Model Architecture

The model is a **Convolutional Neural Network (CNN)** that includes:
- **Conv2D** and **MaxPooling2D** layers for feature extraction.
- **Fully Connected** layers for classification.
- **Dropout** layer to prevent overfitting.

## Results

After training, the model achieves **~95% accuracy** on the test set.

---
