# TalkToHand: Real-Time Hand Gesture Recognition & ASL Translation

**TalkToHand** is a real-time hand gesture recognition system using [MediaPipe](https://mediapipe.dev/) and a custom neural network built with TensorFlow/Keras. It captures hand landmarks, trains on American Sign Language (ASL) gestures, and translates them into readable text.

---

## 🧠 Features <br>

- Collect labeled hand gesture data using webcam
- Automatically track hand landmarks with MediaPipe
- Train a lightweight CNN model for gesture classification
- Real-time prediction and ASL letter translation from webcam input
- Save final text output to file

---

## 🚀 Getting Started <br>
Clone the Repository typing the commands below in your terminal:
```bash
git clone https://github.com/bmestref/TalkToHand.git
cd TalkToHand
```
Also, make sure the following dependencies are installed properly:
```
pip install opencv-python mediapipe numpy pandas tensorflow

```

## 🎯 Collecting Data <br>
To collect labeled ASL gesture data:
```
from data_collection import get_data

get_data(
    target_dir='<your/target_folder>',
    name='<name_file>',
    auto_mode=True,
    limit=<num_of_samples>,
    letter='<letter>',
    speed=1
)
```
