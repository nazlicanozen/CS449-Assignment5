import cv2
import mediapipe as mp
import random
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

STATE_MENU = 0
STATE_GAME = 1
STATE_SET_DIFFICULTY = 2
STATE_EXIT = 3

current_state = STATE_MENU
previous_state = STATE_MENU

menu_options = ["Play", "Set Difficulty", "Exit"]

dots = []
score = 0
game_duration = 60
start_time = time.time()

cap = cv2.VideoCapture(0)

# Effects
caught_effects = []
missed_effects = []

# Hover/Selection Variables
hovered_option = None
hover_start_time = 0
hover_duration = 1.0  # 1 second to select

# Back arrow variables
back_hover_start = 0
back_hovered = False
arrow_text = "<-"
arrow_x = 50
arrow_y = 50

# Dot colors
dot_colors = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (0, 255, 255)   # Yellow
]

# Difficulty handling
difficulty = 1
line_y = 150
line_margin = 50
thumb_hover_side = None
thumb_hover_start = 0

def map_difficulty_to_x(difficulty, width):
    line_start_x = line_margin
    line_end_x = width - line_margin
    intervals = 9
    return int(line_start_x + (difficulty - 1) * (line_end_x - line_start_x) / intervals)

def show_difficulty_line(frame, width, height):
    line_start_x = line_margin
    line_end_x = width - line_margin
    # Draw the main line
    cv2.line(frame, (line_start_x, line_y), (line_end_x, line_y), (255,255,255), 2)
    # Draw the difficulty marker
    diff_x = map_difficulty_to_x(difficulty, width)
    cv2.circle(frame, (diff_x, line_y), 10, (255,255,255), -1)

    # Display numeric scales
    cv2.putText(frame, "1", (line_start_x-10, line_y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.putText(frame, "10", (line_end_x-20, line_y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

    # Instructions
    cv2.putText(frame, "Point arrow to go back", (50,100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Current difficulty at bottom
    diff_text = f"Current Difficulty: {difficulty}"
    (diff_w, diff_h), _ = cv2.getTextSize(diff_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(frame, diff_text, (width//2 - diff_w//2, height - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

def draw_back_arrow(frame):
    cv2.putText(frame, arrow_text, (arrow_x, arrow_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)

def check_back_arrow_hover(landmarks, width, height):
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    idx_x, idx_y = int(index_tip.x * width), int(index_tip.y * height)
    if (arrow_x - 20 < idx_x < arrow_x + 40) and (arrow_y - 20 < idx_y < arrow_y + 20):
        return True
    return False

def is_pinch_detected(landmarks, width, height):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip_x, thumb_tip_y = int(thumb_tip.x * width), int(thumb_tip.y * height)
    index_tip_x, index_tip_y = int(index_tip.x * width), int(index_tip.y * height)
    distance = ((thumb_tip_x - index_tip_x)**2 + (thumb_tip_y - index_tip_y)**2)**0.5
    return distance < 25

def is_hand_flat(landmarks, width, height):
    threshold = 30
    y_thumb = landmarks[mp_hands.HandLandmark.THUMB_TIP].y * height
    y_index = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height
    y_middle = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * height
    y_ring = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y * height
    y_pinky = landmarks[mp_hands.HandLandmark.PINKY_TIP].y * height
    return max(abs(y_thumb - y_index), abs(y_index - y_middle),
               abs(y_middle - y_ring), abs(y_ring - y_pinky)) < threshold

def detect_menu_selection_by_index(landmarks, width, height):
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    idx_x, idx_y = int(index_tip.x * width), int(index_tip.y * height)
    start_y = height//2
    for i, opt in enumerate(menu_options):
        text_x = width//2 - 100
        text_y = start_y + i*60
        if abs(idx_x - text_x) < 100 and abs(idx_y - text_y) < 30:
            return i
    return None

def check_thumb_direction(landmarks, width, height):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_x = int(thumb_tip.x * width)
    diff_x = map_difficulty_to_x(difficulty, width)

    if thumb_x < diff_x - 40:
        return "left"
    elif thumb_x > diff_x + 40:
        return "right"
    else:
        return "neutral"

def draw_hover_progress(frame, start_time, duration, x, y, text, font_scale=1.5, thickness=3, color=(0,255,0)):
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    elapsed = time.time() - start_time
    progress = min(1.0, elapsed / duration)
    fill_w = int(text_w * progress)
    cv2.rectangle(frame, (x, y - text_h), (x + fill_w, y), color, -1)


while True:
    success, frame = cap.read()
    if not success:
        break

    if current_state != previous_state:
        previous_state = current_state

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        results = hands.process(frame_rgb)

    now = time.time()
    # Update and draw caught effects
    for eff in caught_effects[:]:
        elapsed = now - eff['start']
        if elapsed < 0.5:
            radius = int(30 - elapsed * 60)
            cv2.circle(frame, (eff['x'], eff['y']), radius, (0, 255, 255), 3)
        else:
            caught_effects.remove(eff)

    # Update and draw missed effects
    for eff in missed_effects[:]:
        elapsed = now - eff['start']
        if elapsed < 0.5:
            w = int(10 + (elapsed * 100))
            cv2.line(frame, (eff['x'] - w//2, height - 5), (eff['x'] + w//2, height - 5), (0, 0, 255), 2)
        else:
            missed_effects.remove(eff)

    if current_state == STATE_MENU:
        cv2.putText(frame, "CATCHER", (width//2 - 150, height//2 - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
        start_y = height//2
        for i, opt in enumerate(menu_options):
            cv2.putText(frame, opt, (width//2 - 100, start_y + i*60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                selected = detect_menu_selection_by_index(hand_landmarks.landmark, width, height)
                if selected is not None:
                    if hovered_option == selected:
                        opt = menu_options[selected]
                        text_x = width//2 - 100
                        text_y = start_y + selected*60
                        draw_hover_progress(frame, hover_start_time, hover_duration, text_x, text_y, opt)
                        if time.time() - hover_start_time > hover_duration:
                            if selected == 0: # Play
                                current_state = STATE_GAME
                                start_time = time.time()
                                score = 0
                                dots = []
                            elif selected == 1: # Set Difficulty
                                current_state = STATE_SET_DIFFICULTY
                            elif selected == 2: # Exit
                                current_state = STATE_EXIT
                            hovered_option = None
                            hover_start_time = 0
                    else:
                        hovered_option = selected
                        hover_start_time = time.time()
                else:
                    hovered_option = None
                    hover_start_time = 0

                # Visualization of fingertip
                idx_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
                idx_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
                cv2.circle(frame, (idx_x, idx_y), 10, (255, 0, 0), -1)

    elif current_state == STATE_SET_DIFFICULTY:
        show_difficulty_line(frame, width, height)
        draw_back_arrow(frame)

        # Arrow highlight if hovered
        if back_hovered:
            draw_hover_progress(frame, back_hover_start, hover_duration, arrow_x, arrow_y, arrow_text, font_scale=1.5, thickness=3)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check back arrow hover
                if check_back_arrow_hover(hand_landmarks.landmark, width, height):
                    if not back_hovered:
                        back_hovered = True
                        back_hover_start = time.time()
                    else:
                        if time.time() - back_hover_start > hover_duration:
                            current_state = STATE_MENU
                            back_hovered = False
                            back_hover_start = 0
                else:
                    back_hovered = False
                    back_hover_start = 0

                # Check thumb direction for difficulty ONLY if intersection glow is present
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_x = int(thumb_tip.x * width)
                thumb_y = int(thumb_tip.y * height)
                line_start_x = line_margin
                line_end_x = width - line_margin
                intersection_condition = (abs(thumb_y - line_y) < 20 and line_start_x <= thumb_x <= line_end_x)

                # Intersection glow if thumb close
                if intersection_condition:
                    cv2.circle(frame, (thumb_x, line_y), 15, (0,255,0), 2)

                direction = check_thumb_direction(hand_landmarks.landmark, width, height)
                # Only change difficulty if intersection condition is met
                if intersection_condition:
                    if direction == "left":
                        if thumb_hover_side != "left":
                            thumb_hover_side = "left"
                            thumb_hover_start = time.time()
                        else:
                            if time.time() - thumb_hover_start > 1.0:
                                if difficulty > 1:
                                    difficulty -= 1
                                thumb_hover_side = None
                                thumb_hover_start = 0
                    elif direction == "right":
                        if thumb_hover_side != "right":
                            thumb_hover_side = "right"
                            thumb_hover_start = time.time()
                        else:
                            if time.time() - thumb_hover_start > 1.0:
                                if difficulty < 10:
                                    difficulty += 1
                                thumb_hover_side = None
                                thumb_hover_start = 0
                    else:
                        thumb_hover_side = None
                        thumb_hover_start = 0
                else:
                    # If not intersecting, no changes
                    thumb_hover_side = None
                    thumb_hover_start = 0

    elif current_state == STATE_GAME:
        flat_hand_detected = False
        finger_x, finger_y = 0, 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if is_hand_flat(hand_landmarks.landmark, width, height):
                    flat_hand_detected = True
                    finger_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * width)
                    finger_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * height)

                    rect_width = 120
                    rect_height = 20
                    cv2.rectangle(frame,
                                  (finger_x - rect_width//2, finger_y - rect_height//2),
                                  (finger_x + rect_width//2, finger_y + rect_height//2),
                                  (173,255,47), -1)
                    cv2.putText(frame, "Collect dots!", (finger_x + rect_width//2 + 10, finger_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        speed = difficulty * 2
        frequency = max(20 - difficulty, 5)
        size = max(10, 20 - difficulty)

        elapsed_time = time.time() - start_time
        remaining_time = max(0, game_duration - elapsed_time)

        time_color = (255,255,255)
        if remaining_time <= 10:
            time_color = (0,0,255)

        if remaining_time > 0:
            # Show Score and Dif at top as usual
            cv2.putText(frame, f"Time: {int(remaining_time)}s", (width - 150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, time_color, 2)
            cv2.putText(frame, f"Score: {score}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Dif: {difficulty}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        else:
            # Game over
            cv2.putText(frame, "GAME OVER", (width // 2 - 100, height // 2 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.putText(frame, f"Final Score: {score}", (width // 2 - 120, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Point arrow to go back", 
                        (width // 2 - 150, height // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            draw_back_arrow(frame)
            # On game over, score and dif labels should appear at bottom
            # Remove top-left appearance; show them at bottom
            bottom_text = f"Score: {score}   Dif: {difficulty}"
            (b_w, b_h), _ = cv2.getTextSize(bottom_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.putText(frame, bottom_text, (width//2 - b_w//2, height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # Arrow highlight if hovered
            if back_hovered:
                draw_hover_progress(frame, back_hover_start, hover_duration, arrow_x, arrow_y, arrow_text, font_scale=1.5, thickness=3)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if check_back_arrow_hover(hand_landmarks.landmark, width, height):
                        if not back_hovered:
                            back_hovered = True
                            back_hover_start = time.time()
                        else:
                            if time.time() - back_hover_start > hover_duration:
                                current_state = STATE_MENU
                                back_hovered = False
                                back_hover_start = 0
                    else:
                        back_hovered = False
                        back_hover_start = 0

        if remaining_time > 0:
            # Normal gameplay logic
            for dot in dots[:]:
                dot['y'] += speed
                cv2.circle(frame, (dot['x'], dot['y']), size, dot['color'], -1)

                if flat_hand_detected:
                    rect_width = 120
                    rect_height = 20
                    if (finger_x - rect_width//2 <= dot['x'] <= finger_x + rect_width//2) and \
                       (finger_y - rect_height//2 <= dot['y'] <= finger_y + rect_height//2):
                        caught_effects.append({'x': dot['x'], 'y': dot['y'], 'start': time.time()})
                        dots.remove(dot)
                        score += 1
                        continue

                if dot['y'] > height:
                    missed_effects.append({'x': dot['x'], 'start': time.time()})
                    dots.remove(dot)

            if random.randint(1, frequency) == 1:
                color_choice = random.choice(dot_colors)
                dots.append({'x': random.randint(50, width - 50), 'y': 0, 'color': color_choice})

    elif current_state == STATE_EXIT:
        break

    key = cv2.waitKey(5)
    if key & 0xFF == ord('q'):
        if current_state == STATE_MENU:
            break
        elif current_state == STATE_SET_DIFFICULTY:
            current_state = STATE_MENU
        elif current_state == STATE_GAME:
            current_state = STATE_MENU

    cv2.imshow("Catcher", frame)

cap.release()
cv2.destroyAllWindows()
