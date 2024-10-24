import cv2
import mediapipe as mp
import numpy as np
import os

# Ask user for video input and full screen preference
use_video_input = input("Do you want to use a video file? (y/n): ").strip().lower() == 'y'
fullscreen = input("Do you want to use full screen? (y/n): ").strip().lower() == 'y'

# Ask if the user wants to see landmarks
show_landmarks = input("Do you want to see the landmarks? (y/n): ").strip().lower() == 'y'

# Open video capture
if use_video_input:
    video_path = 'sample.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
else:
    cap = cv2.VideoCapture(0)  # Default to webcam

# Check if the camera opened successfully
if not cap.isOpened():
    raise ValueError("Error: Unable to open video source")

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load clothing images from the 'shirts' and 'pants' directories
def load_clothing(directory):
    return [cv2.imread(os.path.join(directory, img), cv2.IMREAD_UNCHANGED) for img in os.listdir(directory)]

shirts = load_clothing('shirts')
pants = load_clothing('pants')
current_shirt = 0
current_pants = 0

# Overlay clothes function
def overlay_clothes(frame, img, points):
    h, w = img.shape[:2]
    
    if img.shape[2] != 4:
        raise ValueError("Input image does not have 4 channels (RGBA) for transparency.")

    pts_dst = np.array(points, dtype=np.float32)
    pts_src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped_img = cv2.warpPerspective(img, matrix, (frame.shape[1], frame.shape[0]))

    mask = warped_img[..., 3] / 255.0
    mask_inv = 1.0 - mask

    for c in range(3):
        frame[..., c] = (mask * warped_img[..., c] + mask_inv * frame[..., c])

    return frame

# Function to draw translucent buttons
def draw_button(frame, label, pos, size=(150, 40), alpha=0.5):
    x, y = pos
    w, h = size
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), -1)  # White button
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, label, (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

# Button click states
button_clicked = None
screen_state = 'main'  # 'main', 'shirts', 'pants'

def mouse_callback(event, x, y, flags, param):
    global button_clicked, screen_state, current_shirt, current_pants
    button_positions = []

    if screen_state == 'main':
        button_positions = [(10, 10), (10, 60)]  # Buttons for Shirts and Pants
    elif screen_state == 'shirts':
        button_positions = [(10, 10)] + [(10, i * 50 + 60) for i in range(len(shirts))]  # Back and Shirt Buttons
    elif screen_state == 'pants':
        button_positions = [(10, 10)] + [(10, i * 50 + 60) for i in range(len(pants))]  # Back and Pants Buttons

    # Check for clicks within button positions
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (bx, by) in enumerate(button_positions):
            if bx <= x <= bx + 150 and by <= y <= by + 40:
                button_clicked = i
                if screen_state == 'main':
                    if button_clicked == 0:
                        screen_state = 'shirts'
                    elif button_clicked == 1:
                        screen_state = 'pants'
                elif screen_state == 'shirts':
                    if button_clicked == 0:  # Back Button
                        screen_state = 'main'
                    else:
                        current_shirt = button_clicked - 1  # Select shirt
                elif screen_state == 'pants':
                    if button_clicked == 0:  # Back Button
                        screen_state = 'main'
                    else:
                        current_pants = button_clicked - 1  # Select pants


cv2.namedWindow('Virtual Try-On', cv2.WINDOW_NORMAL)
if fullscreen:
    cv2.setWindowProperty('Virtual Try-On', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.setMouseCallback('Virtual Try-On', mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from video/camera")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = np.array([(lm.x, lm.y) for lm in results.pose_landmarks.landmark])

        frame_h, frame_w, _ = frame.shape

        if show_landmarks:
            for i, landmark in enumerate(landmarks):
                x, y = int(landmark[0] * frame_w), int(landmark[1] * frame_h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green dot for each landmark

        # Get body landmark positions
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] * [frame_w, frame_h]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] * [frame_w, frame_h]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value] * [frame_w, frame_h]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value] * [frame_w, frame_h]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value] * [frame_w, frame_h]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value] * [frame_w, frame_h]

        # Clip coordinates to prevent out-of-frame positions
        left_shoulder = np.clip(left_shoulder, [0, 0], [frame_w, frame_h]).astype(int)
        right_shoulder = np.clip(right_shoulder, [0, 0], [frame_w, frame_h]).astype(int)
        left_hip = np.clip(left_hip, [0, 0], [frame_w, frame_h]).astype(int)
        right_hip = np.clip(right_hip, [0, 0], [frame_w, frame_h]).astype(int)
        left_ankle = np.clip(left_ankle, [0, 0], [frame_w, frame_h]).astype(int)
        right_ankle = np.clip(right_ankle, [0, 0], [frame_w, frame_h]).astype(int)

        # Adjust landmarks for clothing overlay
        left_shoulder[1] -= 40
        right_shoulder[1] -= 40
        left_shoulder[0] += 60
        left_hip[0] += 60
        right_shoulder[0] -= 60
        right_hip[0] -= 60

        # Overlay current shirt and pants
        shirt_points = [left_shoulder, right_shoulder, right_hip, left_hip]
        frame = overlay_clothes(frame, shirts[current_shirt], shirt_points)

        pants_points = [left_hip, right_hip, right_ankle, left_ankle]
        frame = overlay_clothes(frame, pants[current_pants], pants_points)

    # Button UI logic
    if screen_state == 'main':
        draw_button(frame, 'Shirts', (10, 10))
        draw_button(frame, 'Pants', (10, 60))
    elif screen_state == 'shirts':
        draw_button(frame, 'Back', (10, 10))
        for i, shirt in enumerate(shirts):
            draw_button(frame, f'Shirt {i + 1}', (10, i * 50 + 60))
    elif screen_state == 'pants':
        draw_button(frame, 'Back', (10, 10))
        for i, pant in enumerate(pants):
            draw_button(frame, f'Pant {i + 1}', (10, i * 50 + 60))

    # Display the frame
    cv2.imshow('Virtual Try-On', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
