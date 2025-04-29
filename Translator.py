import cv2
import mediapipe as mp
import numpy as np
import os
from tensorflow.keras.models import load_model



def display_text(text):
    """
    Displays a string as wrapped multi-line text on a white image using OpenCV.

    This function takes a string of text and renders it onto a blank white image (200x400 px).
    It automatically wraps text to the next line if it exceeds the maximum width of the image.

    Parameters
    ----------
    text : str
        The string to display.

    Returns
    -------
    np.ndarray
        An image (NumPy array) with the rendered multi-line text.
    """
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

# os.chdir(r'C:\Users\Usuario\Desktop\Projects\TalkToHand')
def hand_tracking(target_dir):
    """
    Captures hand gestures via webcam and translates them into text using a trained model.

    This function uses MediaPipe to detect hand landmarks in real-time. It then crops and
    processes the detected hand region, passes it through a pre-trained neural network
    to predict the corresponding ASL character, and appends it to an accumulating string.
    The final text is displayed live and saved to a file when the session ends.

    The model is only activated when the '+' key is pressed, and predictions are made
    every 5 frames to reduce redundancy.

    Parameters
    ----------
    target_dir : str
        Directory path where the output 'translation.txt' will be saved.

    Requirements
    ------------
    - The Keras model must be located at 'models/RightHandModel.keras'.
    - Webcam access must be available.
    - Press 'ESC' to terminate the session and save the translation.
    - Press '+' to enable text accumulation from predictions.
    """
    
    labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
                    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L',
                      11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q',
                        16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}

    text = ''

    flag = False
    # Load prediction model
    model = load_model('models/RightHandModel.keras')

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

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        full_frame = frame.copy()
        hand_crop = None

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(full_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, c = frame.shape
                x_list, y_list = [], []

                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_list.append(x)
                    y_list.append(y)

                x_mean = int(np.mean(x_list))
                y_mean = int(np.mean(y_list))
                y_mean += 40  # Small vertical shift

                padding = 175
                x_min = max(x_mean - padding, 0)
                y_min = max(y_mean - padding, 0)
                x_max = min(x_mean + padding, w)
                y_max = min(y_mean + padding, h)

                hand_crop = frame[y_min:y_max, x_min:x_max]

                if hand_crop.size != 0:
                    hand_crop = cv2.resize(hand_crop, (350, 350))

                    for lm in hand_landmarks.landmark:
                        x = int((lm.x * w - x_min) * (350 / (x_max - x_min)))
                        y = int((lm.y * h - y_min) * (350 / (y_max - y_min)))
                        cv2.circle(hand_crop, (x, y), 4, (0, 255, 0), -1)

        # Display frames
        cv2.imshow("Full Frame", full_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('+'):
            flag = True

        if hand_crop is not None:
            cv2.imshow("Hand Crop", hand_crop)
            if flag == True:
                if frame_count % 5 == 0:
                    row_data = []
                    for lm in hand_landmarks.landmark:
                        x = int((lm.x * w - x_min) * (350 / (x_max - x_min))) / 350
                        y = int((lm.y * h - y_min) * (350 / (y_max - y_min))) / 350
                        row_data.append(x)
                        row_data.append(y)
                        
                    prediction = model(np.array(row_data).reshape(1, -1), training=False).numpy()
                    y_pred = np.argmax(np.array(prediction).reshape(1,-1), axis=1)[0]
                    
                    predicted_character = labels_dict[y_pred]

                    if prediction[0][y_pred] >= 0.95:
                        if len(text) == 0 or text[-1] != predicted_character:
                            text += predicted_character
        else:
            blank = 255 * np.ones(shape=[350, 350, 3], dtype=np.uint8)
            cv2.imshow("Hand Crop", blank)


        text_window = display_text(text)
                        
        # Display the text window
        cv2.imshow("Text Output", text_window)

        
        if key == 27:
            with open(os.path.join(target_dir, 'translation.txt'), "w") as text_file:
                text_file.write(text)
            break

        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

#hand_tracking(target_dir = r'C:\Users\Usuario\Desktop\Projects\TalkToHand')
