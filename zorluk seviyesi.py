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

        # Get difficulty level from slider
        difficulty = cv2.getTrackbarPos("Difficulty", "Falling Colourful Dots")
        speed = difficulty * 2  # Increase speed based on difficulty (1-10)
        frequency = max(20 - difficulty, 5)  # Lower number = more frequent dots (adjust as needed)
        size = max(10, 20 - difficulty)  # Smaller dots for harder difficulty

        # Calculate remaining time
        game_active = True
        elapsed_time = time.time() - start_time
        remaining_time = max(0, game_duration - elapsed_time)

        # Display chronometer
        time_text = f"Time: {int(remaining_time)}s"
        (text_width, text_height), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        right_top_x = width - text_width - 10
        right_top_y = text_height + 10  # Adjusted to stay at the top

        cv2.putText(
            frame,
            time_text,
            (right_top_x, right_top_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255) if remaining_time > 10 else (0, 0, 255),
            2,
        )

        if remaining_time <= 0:
            game_active = False

        # Update game dots
        if game_active:
            for dot in dots[:]:
                dot['y'] += speed  # Dots fall faster with higher difficulty
                cv2.circle(frame, (dot['x'], dot['y']), size, (0, 255, 0), -1)  # Draw dots

                # Remove dot if it reaches the bottom
                if dot['y'] > height:
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
