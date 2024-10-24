import cv2
import mediapipe as mp
import numpy as np

# Set up camera or video
useVideo = True  # Change to False to use a webcam

if useVideo:
    video_path = 'sample.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
else:
    cap = cv2.VideoCapture(0)

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load T-shirt and pants images (ensure these images have transparency, i.e., PNG format)
tshirt_img = cv2.imread('shirt.png', cv2.IMREAD_UNCHANGED)
pants_img = cv2.imread('pants.png', cv2.IMREAD_UNCHANGED)

# Check if the images were loaded correctly
if tshirt_img is None or pants_img is None:
    raise ValueError("Error loading 'tshirt.png' or 'pants.png'. Check the file paths.")

def overlay_image(background, overlay, x, y, overlay_size=None):
    """
    Optimized overlay function using NumPy for faster operations.
    """
    if overlay_size:
        overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_LINEAR)

    h, w, _ = overlay.shape
    bg_h, bg_w, _ = background.shape

    # Ensure overlay is within frame bounds
    if x < 0 or y < 0 or x + w > bg_w or y + h > bg_h:
        return background

    # Extract the alpha mask of the overlay image
    alpha_mask = overlay[..., 3] / 255.0
    alpha_inv = 1.0 - alpha_mask

    # Apply alpha blending using NumPy slicing
    for c in range(3):  # Iterate over color channels
        background[y:y + h, x:x + w, c] = (alpha_mask * overlay[..., c] +
                                           alpha_inv * background[y:y + h, x:x + w, c])

    return background

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with the Pose model
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Extract landmarks using NumPy arrays
        landmarks = np.array([(lm.x, lm.y) for lm in results.pose_landmarks.landmark])

        frame_h, frame_w, _ = frame.shape
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] * [frame_w, frame_h]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] * [frame_w, frame_h]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value] * [frame_w, frame_h]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value] * [frame_w, frame_h]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value] * [frame_w, frame_h]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value] * [frame_w, frame_h]

        # Calculate shoulder width using NumPy
        shoulder_width = int(np.linalg.norm(left_shoulder - right_shoulder)) + 80
        hip_width = int(np.linalg.norm(left_hip - right_hip))

        midpoint_shoulder = np.mean([left_shoulder, right_shoulder], axis=0).astype(int)
        midpoint_hip = np.mean([left_hip, right_hip], axis=0).astype(int)
        midpoint_ankle = np.mean([left_ankle, right_ankle], axis=0).astype(int)

        # Overlay the T-shirt
        x_pos_tshirt = int(midpoint_shoulder[0] - shoulder_width // 2)
        y_pos_tshirt = int(midpoint_shoulder[1] - shoulder_width // 4)
        tshirt_height = shoulder_width + 50

        frame = overlay_image(frame, tshirt_img, x_pos_tshirt, y_pos_tshirt,
                              (shoulder_width, tshirt_height))

        # Calculate the pants height (distance from hips to feet)
        pants_height = int(midpoint_ankle[1] - midpoint_hip[1])

        # Adjust pants width to be wider
        adjusted_pants_width = int(hip_width * 2)  # Increase width adjustment factor to 1.5

        # Make sure adjusted_pants_width and pants_height are positive
        if adjusted_pants_width > 0 and pants_height > 0:
            # Pants position and size
            x_pos_pants = int(midpoint_hip[0] - adjusted_pants_width // 2)  # Center horizontally
            y_pos_pants = int(midpoint_hip[1])  # Adjust the vertical position at hips
            frame = overlay_image(frame, pants_img, x_pos_pants, y_pos_pants,
                                  (adjusted_pants_width, pants_height))

    # Display the frame
    cv2.imshow('Virtual Try-On', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
