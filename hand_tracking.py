
import cv2
import mediapipe as mp

def hand_tracking_model():
    # Initialize MediaPipe Hands module
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Set up webcam capture
    cap = cv2.VideoCapture(0)  # Open the webcam

    # Initialize the hand detection model
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while cap.isOpened():
        # Read the frame from the webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Convert the frame to RGB (OpenCV uses BGR by default)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand landmarks
        results = hands.process(rgb_frame)

        # If hands are detected, draw landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the frame with landmarks
        cv2.imshow("Hand Detection", frame)
        
        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# hand_tracking_model()
