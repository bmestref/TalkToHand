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
    speed=1 # The bigger the faster
)
```
Controls: <br>

- Press + to start collecting frames (auto mode). If auto_mode = False, first press the letter you want to record and then + to save it.

- Press ESC to stop and save the CSV.

## 🧠 Model Architecture <br>
```
def HandTranslate(num_classes=3):
    model = Sequential([
        Input(shape=(42,)),
        Reshape((21, 2)),
        Conv1D(32, 3, padding='same', activation='relu'),
        MaxPooling1D(2),
        Conv1D(64, 3, padding='same', activation='relu'),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model
```

## 🎥 Real-Time Gesture Translation <br>
Run the translator with:
```
from main import hand_tracking

hand_tracking(target_dir='<target_folder_to_store_txt')
```
Controls: <br>

- Press + to activate prediction mode

- Press ESC to save the translated text to translation.txt

## 🔧 Future Improvements <br>
 - Expand dataset for more ASL letters and gestures

 - Add dynamic (motion-based) gesture support

 - Optimize with TensorFlow Lite for mobile or embedded deployment

 - Improve prediction speed and model accuracy
