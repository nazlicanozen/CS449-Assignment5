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

# Function to detect pinch gesture (for difficulty adjustment)
def is_pinch_detected(landmarks, width, height):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    thumb_tip_x, thumb_tip_y = int(thumb_tip.x * width), int(thumb_tip.y * height)
    index_tip_x, index_tip_y = int(index_tip.x * width), int(index_tip.y * height)

    distance = ((thumb_tip_x - index_tip_x)**2 + (thumb_tip_y - index_tip_y)**2)**0.5
    return distance < 25  # Pinch detected if the distance is <30 pixels

# Function to check if the hand is flat (for collecting dots)
def is_hand_flat(landmarks, width, height):
    y_thumb = landmarks[mp_hands.HandLandmark.THUMB_TIP].y * height
    y_index = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height
    y_middle = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * height
    y_ring = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y * height
    y_pinky = landmarks[mp_hands.HandLandmark.PINKY_TIP].y * height

    return max(abs(y_thumb - y_index), abs(y_index - y_middle),
               abs(y_middle - y_ring), abs(y_ring - y_pinky)) < 20

# Initialize OpenCV window and trackbar
cv2.namedWindow("Falling Colourful Dots")
cv2.createTrackbar("Difficulty", "Falling Colourful Dots", 1, 10, lambda x: None)

# Start MediaPipe Hands
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    last_pinch_time = 0  # To avoid rapid difficulty change
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        height, width, _ = frame.shape

        flat_hand_detected = False
        hand_x, hand_y = 0, 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Handle pinch gesture for difficulty adjustment
                if is_pinch_detected(hand_landmarks.landmark, width, height):
                    current_time = time.time()
                    if current_time - last_pinch_time > 0.5:  # 0.5 seconds debounce time
                        current_difficulty = cv2.getTrackbarPos("Difficulty", "Falling Colourful Dots")
                        new_difficulty = (current_difficulty % 10) + 1
                        cv2.setTrackbarPos("Difficulty", "Falling Colourful Dots", new_difficulty)
                        last_pinch_time = current_time

                # Check if the hand is flat for collecting dots
                if is_hand_flat(hand_landmarks.landmark, width, height):
                    flat_hand_detected = True
                    hand_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width)
                    hand_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height)
                    cv2.circle(frame, (hand_x, hand_y), 20, (173, 255, 47), -1)
                    cv2.putText(frame, "Collect dots!", (hand_x + 30, hand_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Game logic
        difficulty = cv2.getTrackbarPos("Difficulty", "Falling Colourful Dots")
        speed = difficulty * 2
        frequency = max(20 - difficulty, 5)
        size = max(10, 20 - difficulty)

        elapsed_time = time.time() - start_time
        remaining_time = max(0, game_duration - elapsed_time)

        cv2.putText(frame, f"Time: {int(remaining_time)}s", (width - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if remaining_time <= 10:  
            cv2.putText(frame, f"Time: {int(remaining_time)}s", (width - 150, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  

        if remaining_time <= 0:
            cv2.putText(frame, "GAME OVER", (width // 2 - 100, height // 2 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.putText(frame, f"Final Score: {score}", (width // 2 - 120, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Press [q] to exit", (width // 2 - 150, height // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)      

        for dot in dots[:]:
            dot['y'] += speed
            cv2.circle(frame, (dot['x'], dot['y']), size, (0, 255, 0), -1)

            if flat_hand_detected and abs(dot['x'] - hand_x) < 30 and abs(dot['y'] - hand_y) < 30:
                dots.remove(dot)
                score += 1

            elif dot['y'] > height:
                dots.remove(dot)

        if random.randint(1, frequency) == 1:
            dots.append({'x': random.randint(50, width - 50), 'y': 0})

        cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Falling Colourful Dots', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
