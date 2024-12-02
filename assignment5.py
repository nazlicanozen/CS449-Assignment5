import cv2
import mediapipe as mp
import random

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize game variables
cap = cv2.VideoCapture(0)
dots = []  # List to store falling dots
score = 0

# Function to check if the hand is flat
# The dots will only be collectible when the hand is flat
# To collect the dots, the user should show the flat hand to the camera
def is_hand_flat(landmarks, width, height):
    """
    Check if the hand is flat based on the y-coordinates of the fingertips.
    """
    # Get y-coordinates of the fingertips
    y_thumb = landmarks[mp_hands.HandLandmark.THUMB_TIP].y * height
    y_index = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height
    y_middle = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * height
    y_ring = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y * height
    y_pinky = landmarks[mp_hands.HandLandmark.PINKY_TIP].y * height

    # Check if the difference between all y-coordinates is small (threshold: 20 pixels)
    # If the y coords are almost the same, it means that the hand is flat
    return max(abs(y_thumb - y_index), abs(y_index - y_middle),
               abs(y_middle - y_ring), abs(y_ring - y_pinky)) < 20

# Start MediaPipe Hands
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip frame for a mirror effect
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame for hand detection
        results = hands.process(frame_rgb)
        height, width, _ = frame.shape

        # Draw hand landmarks and detect a flat hand
        flat_hand_detected = False
        hand_x, hand_y = 0, 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check if the hand is flat
                if is_hand_flat(hand_landmarks.landmark, width, height):
                    flat_hand_detected = True
                    # Use the wrist position to detect dot collection
                    hand_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width)
                    hand_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height)
                    # Draw a marker for the wrist position
                    cv2.circle(frame, (hand_x, hand_y), 20, (173,255,47), -1) 
                    # Add label "Collect dots!" near the wrist
                    cv2.putText(frame, "Collect dots!", (hand_x + 30, hand_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) 

        # Update game dots
        for dot in dots[:]:
            dot['y'] += 5  # Dots fall down
            cv2.circle(frame, (dot['x'], dot['y']), 10, (0, 255, 0), -1)  # Draw dots

            # Check for collision with a flat hand
            if flat_hand_detected and abs(dot['x'] - hand_x) < 30 and abs(dot['y'] - hand_y) < 30:
                dots.remove(dot)
                score += 1  # Increase score

            # Remove dot if it reaches the bottom
            elif dot['y'] > height:
                dots.remove(dot)

        # Generate new dotsrandomly
        if random.randint(1, 20) == 1:  # Adjust the frequency
            dots.append({'x': random.randint(50, width - 50), 'y': 0})

        # Display score
        cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the frame (game window)
        cv2.imshow('Falling Colourful Dots', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
