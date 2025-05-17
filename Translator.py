import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque
import pickle
import time
import tensorflow as tf
from spellchecker import SpellChecker

# Tareas:
# - Mejorar la velocidad con la que el modelo predice
# - Mejorar los métodos de predicción de i,j y d,z tal que si los 3 siguientes reconocimientos no son i ni d entonces sal del buffer

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
# model = tf.keras.models.load_model('models/RightHandModel.keras')

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# # # Save the TFLite model
# with open('models/RightHandModel.tflite', 'wb') as f:
#     f.write(tflite_model)



def extract_landmarks(hand_landmarks):
    return np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])



def hand_tracking(target_dir, autocorrect=False, draw_hands = False):

    spell = SpellChecker() if autocorrect else None
    recording_enabled = False 

    labels_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'delete', 5: 'e', 6: 'f', 7: 'g',
                    8: 'h', 9: 'i', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p',
                      16: 'q', 17: 'r', 18: 's', 19: 'space', 20: 'startstop', 21: 't', 22: 'u',
                        23: 'v', 24: 'w', 25: 'x', 26: 'y'}

    text = ''
    interpreter = tf.lite.Interpreter(model_path='models/RightHandModel.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    def predict_static(input_array):
        interpreter.set_tensor(input_details[0]['index'], input_array.astype(np.float32))
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])

    with open('models/I_J_model.pkl', 'rb') as f:
        model_j = pickle.load(f)
        
    with open('models/D_Z_model.pkl', 'rb') as f:
        model_z = pickle.load(f)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    frame_count = 0
    prediction_buffer = []
    sequence_buffer = deque(maxlen=14)
    delete_buffer = []
    delete_trigger_threshold = 3

    max_buffer_length = 3
    last_added_char = ''
    repeat_count = 0
    last_toggle_time = time.time()
    cooldown_seconds = 1
    motion_step = False
    current_motion_letter = ''
    motion_wait_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        hand_label = None
        is_left_hand = False

        if result.multi_handedness:
            hand_label = result.multi_handedness[0].classification[0].label
            is_left_hand = hand_label == 'Right'

        hand_crop = None

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                if draw_hands:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                x_list, y_list = [], []

                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_list.append(x)
                    y_list.append(y)

                x_mean = int(np.mean(x_list))
                y_mean = int(np.mean(y_list)) + 40

                padding = 175
                x_min = max(x_mean - padding, 0)
                y_min = max(y_mean - padding, 0)
                x_max = min(x_mean + padding, w)
                y_max = min(y_mean + padding, h)

                hand_crop = frame[y_min:y_max, x_min:x_max]

                if hand_crop.size != 0:
                    hand_crop = cv2.resize(hand_crop, (350, 350))

        cv2.imshow("Full Frame", frame)
        key = cv2.waitKey(5) & 0xFF

        if hand_crop is not None:
            if frame_count % 3 == 0:
                row_data = []
                if not motion_step:
                    coords = np.array(extract_landmarks(hand_landmarks))
                    coords[:, 0] = ((coords[:, 0] * w - x_min) * (350 / (x_max - x_min))) / 350
                    coords[:, 1] = ((coords[:, 1] * h - y_min) * (350 / (y_max - y_min))) / 350

                    if is_left_hand:
                        coords[:, 0] = 1.0 - coords[:, 0]

                    row_data = coords.flatten().tolist()

                elif motion_step:
                    coords = np.array(extract_landmarks(hand_landmarks))
                    if is_left_hand:
                        coords[:, 0] = 1.0 - coords[:, 0]

                    coords[:, 0] *= 640
                    coords[:, 1] *= 480

                    row_data = coords.flatten().tolist()

                if motion_step:
                    if current_motion_letter == 'i':
                        sequence_buffer.append([row_data[20], row_data[41]])
                    elif current_motion_letter == 'd':
                        sequence_buffer.append([row_data[8], row_data[29]])

                    if len(sequence_buffer) == 14:
                        dx_dy_features = []
                        for i in range(1, 14):
                            dx = (sequence_buffer[i][0] - sequence_buffer[i - 1][0]) / 640
                            dx_dy_features.append(dx)
                        for i in range(1, 14):
                            dy = (sequence_buffer[i][1] - sequence_buffer[i - 1][1]) / 480
                            dx_dy_features.append(dy)

                        motion_input = np.array(dx_dy_features).reshape(1, -1).astype(np.float32)
                        if current_motion_letter == 'i':
                            is_j = model_j.predict(motion_input)[0]
                            predicted_char = 'j' if is_j == 1 else 'i'
                        elif current_motion_letter == 'd':
                            is_z = model_z.predict(motion_input)[0]
                            predicted_char = 'z' if is_z == 1 else 'd'

                        if recording_enabled:
                            text += predicted_char
                            last_added_char = predicted_char
                            repeat_count = 1

                        motion_step = False
                        current_motion_letter = ''
                        sequence_buffer.clear()

                    continue

                input_arr = np.array(row_data).reshape(1, -1).astype(np.float32)
                prediction = predict_static(input_arr)
                y_pred = np.argmax(prediction)
                confidence = prediction[0][y_pred]

                if confidence >= 0.85:
                    if not motion_step:
                        if labels_dict[y_pred] in ['i', 'd']:
                            motion_step = True
                            current_motion_letter = labels_dict[y_pred]
                            motion_wait_counter = 0
                            print(f"Motion capture started for {current_motion_letter}")
                            prediction_buffer.clear()
                            continue
                    else:
                        if labels_dict[y_pred] not in ['i', 'd']:
                            motion_wait_counter += 1
                            if motion_wait_counter >= 2:
                                print(f"No motion detected, falling back to static '{current_motion_letter}'")
                                predicted_char = current_motion_letter
                                motion_step = False
                                current_motion_letter = ''
                                sequence_buffer.clear()
                                motion_wait_counter = 0

                                if recording_enabled and predicted_char not in ['i', 'd']:
                                    if predicted_char == 'space':
                                        predicted_char = ' '

                                    prediction_buffer.append(predicted_char)
                                    if len(prediction_buffer) > max_buffer_length:
                                        prediction_buffer.pop(0)

                                    recent_preds = prediction_buffer[-max_buffer_length:]
                                    if confidence >= 0.85 or recent_preds.count(predicted_char) >= 1:
                                        if predicted_char != last_added_char:
                                            if predicted_char == ' ':
                                                if autocorrect and spell:
                                                    words = text.strip().split()
                                                    if words:
                                                        corrected = spell.correction(words[-1])
                                                        if corrected:
                                                            words[-1] = corrected
                                                        text = ' '.join(words) + ' '
                                                else:
                                                    text += ' '
                                            else:
                                                text += predicted_char
                                            last_added_char = predicted_char
                                            repeat_count = 1
                                        else:
                                            repeat_count += 1
                                            if repeat_count == 3:
                                                text += predicted_char
                                                repeat_count = 0
                                continue

                    if labels_dict[y_pred] not in ['i','d']:
                        predicted_char = labels_dict[y_pred]

                    if predicted_char == 'startstop':
                        current_time = time.time()
                        if current_time - last_toggle_time > cooldown_seconds:
                            recording_enabled = not recording_enabled
                            print("Recording toggled:", recording_enabled)
                            prediction_buffer.clear()
                            last_added_char = ''
                            repeat_count = 0
                            last_toggle_time = current_time
                        continue

                    if recording_enabled and predicted_char not in ['i', 'd']:
                        if predicted_char == 'delete':
                            delete_buffer.append('delete')
                            if len(delete_buffer) > 4:
                                delete_buffer.pop(0)

                            if delete_buffer.count('delete') >= delete_trigger_threshold:
                                text = text[:-1]
                                print("Deleted last character.")
                                delete_buffer.clear() 
                                text_window = display_text(text)
                                cv2.putText(text_window, "Recording...", (10, 190),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 0), 2)
                                cv2.imshow("Text Output", text_window)
                                continue
                            continue
                        else:
                            delete_buffer.clear() 

                        if predicted_char == 'space':
                            predicted_char = ' ' 

                        prediction_buffer.append(predicted_char)
                        if len(prediction_buffer) > max_buffer_length:
                            prediction_buffer.pop(0)

                        recent_preds = prediction_buffer[-max_buffer_length:]
                        if confidence >= 0.85 or recent_preds.count(predicted_char) >= 1:
                            if predicted_char != last_added_char:
                                if predicted_char == ' ':
                                    if autocorrect and spell:
                                        words = text.strip().split()
                                        if words:
                                            corrected = spell.correction(words[-1])
                                            if corrected:
                                                words[-1] = corrected
                                            text = ' '.join(words) + ' '
                                    else:
                                        text += ' '
                                else:
                                    text += predicted_char
                                last_added_char = predicted_char
                                repeat_count = 1
                            else:
                                repeat_count += 1
                                if repeat_count == 3:
                                    text += predicted_char
                                    repeat_count = 0

        status = "Recording..." if recording_enabled else "Paused"
        text_window = display_text(text)
        cv2.putText(text_window, status, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 0), 2)
        cv2.imshow("Text Output", text_window)

        if key == 27:
            with open(os.path.join(target_dir, 'translation.txt'), "w") as text_file:
                text_file.write(text)
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


hand_tracking(target_dir = r'C:\Users\Usuario\Desktop\Projects\TalkToHand', autocorrect=False, draw_hands = False)
