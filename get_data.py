import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

landmark_names = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip", "middle_mcp",
    "middle_pip", "middle_dip", "middle_tip", "ring_mcp", "ring_pip",
    "ring_dip", "ring_tip", "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
]

landmark_columns = ['Letter']

# Column names for DataFrame
for landmark in landmark_names:
    landmark_columns.append(f'{landmark}_x')
    landmark_columns.append(f'{landmark}_y')

def get_data(target_dir):
    df = pd.DataFrame(columns=landmark_columns)

    # Initialize Mediapipe Hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    # Variable to store the last pressed key
    pressed_key = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for Mediapipe
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
                x_list, y_list = [], []

                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_list.append(x)
                    y_list.append(y)

                x_mean = int(np.mean(np.array(x_list)))
                y_mean = int(np.mean(np.array(y_list)))

                # Move slightly the y_mean below
                y_mean += 40

                # Add some padding around the hand
                padding = 175
                x_min = max(x_mean - padding, 0)
                y_min = max(y_mean - padding, 0)
                x_max = min(x_mean + padding, w)
                y_max = min(y_mean + padding, h)

                # Crop hand from frame
                hand_crop = frame[y_min:y_max, x_min:x_max]

                # Resize cropped hand to 350x350 for consistency
                if hand_crop.size != 0:
                    hand_crop = cv2.resize(hand_crop, (350, 350))

                    # Draw landmarks on hand crop
                    for lm in hand_landmarks.landmark:
                        x = int((lm.x * w - x_min) * (350 / (x_max - x_min)))
                        y = int((lm.y * h - y_min) * (350 / (y_max - y_min)))
                        cv2.circle(hand_crop, (x, y), 4, (0, 255, 0), -1)

        # Show the full frame and hand crop
        cv2.imshow("Full Frame", full_frame)

        if hand_crop is not None:
            cv2.imshow("Hand Crop", hand_crop)
        else:
            # Show black screen if no hand detected
            blank = 255 * np.ones(shape=[350, 350, 3], dtype=np.uint8)
            cv2.imshow("Hand Crop", blank)

        # Handle key press events (one single call to waitKey())
        key = cv2.waitKey(1) & 0xFF

        if key != 255:  # Ignore no-key event
            if key == 27:  # Exclude Esc (27) and 1 (49)
                df.to_csv(os.path.join(target_dir, 'training_data.csv'), index=False)
                print("Data saved to training_data.csv")
                break
            if key == ord('+'):  # If '+' key is pressed, add landmarks to dataframe
                if pressed_key:  # Only if a letter has been pressed
                    row_data = [pressed_key]
                    if hand_crop is not None:
                        for lm in hand_landmarks.landmark:
                            x = int((lm.x * w - x_min) * (350 / (x_max - x_min))) / 350
                            y = int((lm.y * h - y_min) * (350 / (y_max - y_min))) / 350
                            row_data.append(x)
                            row_data.append(y)
                        df.loc[len(df)] = row_data

                        print(f"Data added for {pressed_key}")
            else:  # Store letter pressed (ignore if '+' or Esc are pressed)
                pressed_key = chr(key)  

    cap.release()
    cv2.destroyAllWindows()

get_data(r'C:\Users\Usuario\Desktop\Projects\TalkToHand\data')