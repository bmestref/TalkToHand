import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.ensemble import RandomForestClassifier


def display_text(text):
    text_window = 255 * np.ones(shape=[200, 400, 3], dtype=np.uint8)
    
    # Settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    line_height = 40  # Space between lines
    margin = 10       # Margin from the window edges
    max_width = text_window.shape[1] - 2 * margin

    lines = []
    current_line = ""
    
    for char in text:
        test_line = current_line + char
        text_size, _ = cv2.getTextSize(test_line, font, font_scale, font_thickness)
        text_width = text_size[0]
        
        if text_width < max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = char  # Start new line with current char

    if current_line:
        lines.append(current_line)  # Append last line

    # Draw each line
    y = margin + 30  # Initial Y offset
    for line in lines:
        cv2.putText(text_window, line, (margin, y),
                    font, font_scale, (0, 0, 0), font_thickness)
        y += line_height

    return text_window

def hand_tracking(target_dir):
    labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
                    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L',
                      11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q',
                        16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}

    text = ''

    # Load prediction model
    model = pickle.load('<upload_here_your_model>')

    # Initialize MediaPipe Hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    # Initiate a frame counter to check when to perform the prediction
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        data_aux = []
        x_ = []
        y_ = []


        # frame = cv2.resize(frame, (500, 500))  # Resize to 128x128 if needed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        # Make a copy for full frame display
        full_frame = frame.copy()

        hand_crop = None  # Default if no hand detected


        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks on full frame
                mp_drawing.draw_landmarks(full_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Find bounding box around hand
                h, w, c = frame.shape
                x_min = w
                y_min = h
                x_max = 0
                y_max = 0
                x_list = []
                y_list = []

                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x*w), int(lm.y*h)
                    x_list.append(x)
                    y_list.append(y)
                    
                x_mean = int(np.mean(np.array(x_list)))
                y_mean = int(np.mean(np.array(y_list)))

                # Move slightly the y_mean below
                y_mean +=40

                # Add some padding around the hand
                padding = 175
                x_min = max(x_mean - padding, 0)
                y_min = max(y_mean - padding, 0)
                x_max = min(x_mean + padding, w)
                y_max = min(y_mean + padding, h)

                # Crop hand from frame
                hand_crop = frame[y_min:y_max, x_min:x_max]

                # Resize cropped hand to 128x128 for consistency
                if hand_crop.size != 0:
                    hand_crop = cv2.resize(hand_crop, (350, 350))

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

        cv2.imshow("Full Frame", full_frame)

        if hand_crop is not None:
            cv2.imshow("Hand Crop", hand_crop)
            
            if frame_count % 10 == 0:
                prediction = model.predict([np.asarray(data_aux)])
                probs = model.predict_proba([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                if probs[0][np.argmax(probs[0])] >= 0.95:
                    if len(text) == 0 or text[-1] != predicted_character:
                        text += predicted_character
                        
        else:
                # Show black screen if no hand detected
            blank = 255 * np.ones(shape=[350, 350, 3], dtype=np.uint8)
            cv2.imshow("Hand Crop", blank)

        text_window = display_text(text)

        # Display the text window
        cv2.imshow("Text Output", text_window)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            with open(os.path.join(target_dir, 'translation.txt'), "w") as text_file:
                text_file.write(text)
            break

    cap.release()
    cv2.destroyAllWindows()

hand_tracking()


# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# image = cv2.imread('A_test.jpg')
# image = cv2.resize(image, (224,224))
# image = image / 255.0

# image_crop = np.expand_dims(image, axis = 0)
# prediction = model.predict(image_crop)
# np.argmax(prediction)
# prediction