import cv2
import mediapipe as mp
import random
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize game variables
cap = cv2.VideoCapture(0)
dots = []  # List to store falling dots
score = 0
game_duration = 60  # Game finishes in 60 secs
start_time = time.time()  # Start time for the game

# Function to detect pinch gesture
def is_pinch_detected(landmarks, width, height):
    """
    Check if the pinch gesture is detected based on the distance between thumb and index fingertips.
    """
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    thumb_tip_x, thumb_tip_y = int(thumb_tip.x * width), int(thumb_tip.y * height)
    index_tip_x, index_tip_y = int(index_tip.x * width), int(index_tip.y * height)

    # Calculate Euclidean distance between thumb and index fingertips
    distance = ((thumb_tip_x - index_tip_x)**2 + (thumb_tip_y - index_tip_y)**2)**0.5
    return distance < 30  # Pinch detected if the distance is <30 pixels

# Function to check if the hand is flat
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
    return max(abs(y_thumb - y_index), abs(y_index - y_middle),
               abs(y_middle - y_ring), abs(y_ring - y_pinky)) < 20

# Initialize OpenCV window and trackbar
cv2.namedWindow("Falling Colourful Dots")
cv2.createTrackbar("Difficulty", "Falling Colourful Dots", 1, 10, lambda x: None)  # Slider for difficulty (1-10)

# Start MediaPipe Hands
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    last_pinch_time = 0  # To avoid rapid changes in difficulty
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

        flat_hand_detected = False
        hand_x, hand_y = 0, 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check for pinch gesture
                if is_pinch_detected(hand_landmarks.landmark, width, height):
                    current_time = time.time()
                    if current_time - last_pinch_time > 0.5:  # 0.5 seconds debounce time
                        # Update difficulty (cycle through 1-10)
                        current_difficulty = cv2.getTrackbarPos("Difficulty", "Falling Colourful Dots")
                        new_difficulty = (current_difficulty % 10) + 1  # Increment difficulty, wrap at 10
                        cv2.setTrackbarPos("Difficulty", "Falling Colourful Dots", new_difficulty)
                        last_pinch_time = current_time

                # Check if the hand is flat
                if is_hand_flat(hand_landmarks.landmark, width, height):
                    flat_hand_detected = True
                    # Use the wrist position to detect dot collection
                    hand_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width)
                    hand_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height)
                    # Draw a marker for the wrist position
                    cv2.circle(frame, (hand_x, hand_y), 20, (173, 255, 47), -1)
                    # Add label "Collect dots!" near the wrist
                    cv2.putText(frame, "Collect dots!", (hand_x + 30, hand_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Get difficulty level from slider
        difficulty = cv2.getTrackbarPos("Difficulty", "Falling Colourful Dots")
        speed = difficulty * 2  # Increase speed based on difficulty (1-10)
        frequency = max(20 - difficulty, 5)  # Lower number = more frequent dots
        size = max(10, 20 - difficulty)  # Smaller dots for harder difficulty

        # Calculate remaining time
        elapsed_time = time.time() - start_time
        remaining_time = max(0, game_duration - elapsed_time)

        # Display chronometer
        time_text = f"Time: {int(remaining_time)}s"
        cv2.putText(frame, time_text, (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # End game if time is up
        if remaining_time <= 0:
            break

        # Update game dots
        for dot in dots[:]:
            dot['y'] += speed  # Dots fall faster with higher difficulty
            cv2.circle(frame, (dot['x'], dot['y']), size, (0, 255, 0), -1)  # Draw dots

            # Check for collision with a flat hand
            if flat_hand_detected and abs(dot['x'] - hand_x) < 30 and abs(dot['y'] - hand_y) < 30:
                dots.remove(dot)
                score += 1  # Increase score

            # Remove dot if it reaches the bottom
            elif dot['y'] > height:
                dots.remove(dot)

        # Generate new dots randomly based on frequency
        if random.randint(1, frequency) == 1:
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
