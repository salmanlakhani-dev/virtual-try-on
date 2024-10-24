import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Constants
BUTTON_SIZE = (150, 40)
ALPHA = 0.5  # Button transparency
PINCH_THRESHOLD = 40
PINCH_COOLDOWN = 1  # seconds

# Ask user preferences
use_video_input = input("Use a video file? (y/n): ").strip().lower() == 'y'
fullscreen = input("Use full screen? (y/n): ").strip().lower() == 'y'
show_landmarks = input("Show landmarks? (y/n): ").strip().lower() == 'y'
show_pointer = input("Show pointer? (y/n): ").strip().lower() == 'y'

# Open video source
cap = cv2.VideoCapture('sample.mp4' if use_video_input else 0)

# Error handling if video or camera is not accessible
if not cap.isOpened():
    raise ValueError("Error: Unable to open video source")

# Initialize MediaPipe Pose and Hands solutions
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Function to load clothing images from a directory
def load_clothing(directory):
    return [cv2.imread(os.path.join(directory, img), cv2.IMREAD_UNCHANGED) for img in os.listdir(directory)]

# Load shirt and pants images
shirts = load_clothing('shirts')
pants = load_clothing('pants')

# Overlay clothing onto the frame based on keypoints
def overlay_clothes(frame, img, points):
    if img is None or img.shape[2] != 4:  # Ensure image has alpha channel
        return frame

    h, w = img.shape[:2]
    pts_src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    pts_dst = np.array(points, dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped_img = cv2.warpPerspective(img, matrix, (frame.shape[1], frame.shape[0]))

    # Create mask and blend images
    mask = warped_img[..., 3] / 255.0
    mask_inv = 1.0 - mask

    for c in range(3):
        frame[..., c] = warped_img[..., c] * mask + frame[..., c] * mask_inv
    return frame

# Function to draw translucent buttons
def draw_button(frame, label, pos):
    x, y = pos
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + BUTTON_SIZE[0], y + BUTTON_SIZE[1]), (255, 255, 255), -1)
    cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0, frame)
    cv2.putText(frame, label, (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

# Gesture detection for pinch action
def detect_pinch(index_finger_tip, thumb_tip):
    return np.linalg.norm(np.array(index_finger_tip) - np.array(thumb_tip)) < PINCH_THRESHOLD

# Gesture-based button click detection
def check_gesture_click(finger_tip, button_positions):
    for i, (bx, by) in enumerate(button_positions):
        if bx <= finger_tip[0] <= bx + BUTTON_SIZE[0] and by <= finger_tip[1] <= by + BUTTON_SIZE[1]:
            return i
    return None

# Initialize states
current_shirt = 0
current_pants = 0
screen_state = 'main'
button_clicked = None
last_pinch_time = 0

# Setup fullscreen mode if selected
window_name = 'Virtual Try-On'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
if fullscreen:
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Main loop to process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the video horizontally for proper orientation
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process pose and hand landmarks
    pose_results = pose.process(frame_rgb)
    hand_results = hands.process(frame_rgb)

    frame_h, frame_w = frame.shape[:2]
    landmarks = pose_results.pose_landmarks

    if landmarks:
        # Convert normalized pose landmarks to pixel values
        landmarks = np.array([(lm.x * frame_w, lm.y * frame_h) for lm in landmarks.landmark])

        if show_landmarks:
            for x, y in landmarks:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        # Define keypoints for clothing overlay
        left_shoulder, right_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip, right_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_ankle, right_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Adjust positions slightly for a more natural overlay
        left_shoulder[1] -= 40; right_shoulder[1] -= 40
        left_shoulder[0] += 60; right_shoulder[0] -= 60
        left_hip[0] += 60; right_hip[0] -= 60

        # Overlay current shirt and pants
        shirt_points = [left_shoulder, right_shoulder, right_hip, left_hip]
        frame = overlay_clothes(frame, shirts[current_shirt], shirt_points)

        pants_points = [left_hip, right_hip, right_ankle, left_ankle]
        frame = overlay_clothes(frame, pants[current_pants], pants_points)

    # Button display logic
    button_positions = []
    if screen_state == 'main':
        button_positions = [(10, 10), (10, 60)]
        draw_button(frame, 'Shirts', (10, 10))
        draw_button(frame, 'Pants', (10, 60))
    elif screen_state == 'shirts':
        button_positions = [(10, 10)] + [(10, i * 50 + 60) for i in range(len(shirts))]
        draw_button(frame, 'Back', (10, 10))
        for i in range(len(shirts)):
            draw_button(frame, f'Shirt {i + 1}', (10, i * 50 + 60))
    elif screen_state == 'pants':
        button_positions = [(10, 10)] + [(10, i * 50 + 60) for i in range(len(pants))]
        draw_button(frame, 'Back', (10, 10))
        for i in range(len(pants)):
            draw_button(frame, f'Pant {i + 1}', (10, i * 50 + 60))

    # Hand gesture detection for button interaction
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            x_index, y_index = int(index_tip.x * frame_w), int(index_tip.y * frame_h)
            x_thumb, y_thumb = int(thumb_tip.x * frame_w), int(thumb_tip.y * frame_h)

            if detect_pinch((x_index, y_index), (x_thumb, y_thumb)):
                if show_pointer:
                    cv2.circle(frame, (x_index, y_index), 5, (0, 255, 0), -1)

                current_time = time.time()
                if current_time - last_pinch_time >= PINCH_COOLDOWN:
                    button_clicked = check_gesture_click((x_index, y_index), button_positions)
                    last_pinch_time = current_time
                    if button_clicked is not None:
                        if screen_state == 'main':
                            screen_state = 'shirts' if button_clicked == 0 else 'pants'
                        elif screen_state in ['shirts', 'pants'] and button_clicked == 0:
                            screen_state = 'main'
                        elif screen_state == 'shirts':
                            current_shirt = button_clicked - 1
                        elif screen_state == 'pants':
                            current_pants = button_clicked - 1
            elif show_pointer:
                cv2.circle(frame, (x_index, y_index), 5, (0, 0, 255), -1)

    # Display frame
    cv2.imshow(window_name, frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
