
import cv2
import mediapipe as mp
import numpy as np

def hand_tracking():
    # Initialize MediaPipe Hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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
                    # x_min = min(x_min, x)
                    # y_min = min(y_min, y)
                    # x_max = max(x_max, x)
                    # y_max = max(y_max, y)
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

        # Display both windows
        cv2.imshow("Full Frame", full_frame)

        if hand_crop is not None:
            cv2.imshow("Hand Crop", hand_crop)
        else:
            # Show black screen if no hand detected
            blank = 255 * np.ones(shape=[250, 250, 3], dtype=np.uint8)
            cv2.imshow("Hand Crop", blank)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# hand_tracking()