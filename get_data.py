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

def get_data(target_dir, name, auto_mode, limit=None, letter=None, speed = None):
    df = pd.DataFrame(columns=landmark_columns)
    count_frames = 0
    captured_frames = 0

    # Initialize Mediapipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    pressed_key = ""  # Last pressed key
    start_auto_capture = False  # New flag to control auto-mode start

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

        if hand_crop is not None:
            cv2.imshow("Hand Crop", hand_crop)
        else:
            blank = 255 * np.ones(shape=[350, 350, 3], dtype=np.uint8)
            cv2.imshow("Hand Crop", blank)

        key = cv2.waitKey(1) & 0xFF

        if auto_mode and start_auto_capture and hand_crop is not None:
            if count_frames % speed == 0:
                captured_frames += 1
                row_data = [letter]
                for lm in hand_landmarks.landmark:
                    x = int((lm.x * w - x_min) * (350 / (x_max - x_min))) / 350
                    y = int((lm.y * h - y_min) * (350 / (y_max - y_min))) / 350
                    row_data.append(x)
                    row_data.append(y)
                df.loc[len(df)] = row_data
                print(f"Auto Mode: Data added frame {captured_frames} for letter {letter}")
            

        if key != 255:
            if key == 27:  # ESC key pressed
                df.to_csv(os.path.join(target_dir, f'{name}.csv'), index=False)
                print(f"Data saved to {name}.csv")
                break

            if key == ord('+'):
                if auto_mode:
                    start_auto_capture = True  # Start auto collection after '+' pressed
                    print("Auto mode activated!")
                else:
                    if pressed_key and hand_crop is not None:
                        row_data = [pressed_key]
                        for lm in hand_landmarks.landmark:
                            x = int((lm.x * w - x_min) * (350 / (x_max - x_min))) / 350
                            y = int((lm.y * h - y_min) * (350 / (y_max - y_min))) / 350
                            row_data.append(x)
                            row_data.append(y)
                        df.loc[len(df)] = row_data
                        print(f"Manual mode: Data added for letter {pressed_key}.")

            else:
                # Save the last letter pressed
                pressed_key = chr(key)

        count_frames += 1

        if limit is not None and captured_frames >= limit:
            df.to_csv(os.path.join(target_dir, f'{name}.csv'), index=False)
            print("Reached limit of frames.")
            break

    cap.release()
    cv2.destroyAllWindows()


get_data(r'C:\Users\Usuario\Desktop\Projects\TalkToHand\data\medium_hand', 'G_medium_hand', auto_mode = True, limit = 3000, letter = 'g', speed = 1)

os.chdir(os.path.join(os.getcwd(), "data\medium_hand"))
for i, file in enumerate(os.listdir()):
    if i == 0:
        df = pd.read_csv(file, index_col = 0)
    else:
        df_new = pd.read_csv(file)
        df = pd.concat([df, df_new], axis = 0)

df.to_csv('ABCDEFG_medium_hand.csv')
    