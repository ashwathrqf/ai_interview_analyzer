"""Head pose estimation (looking down/left/right) using MediaPipe FaceMesh
landmarks combined with OpenCV's solvePnP for real yaw/pitch/roll angles.

Approach: we use a small set of stable facial landmarks (nose tip, chin,
eye corners, mouth corners) and a generic 3D face model to solve for the
head's rotation relative to the camera. This gives actual angles in degrees,
not just a heuristic guess from raw 2D landmark positions.
"""

import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Landmark indices for points we use in the pose solve (MediaPipe FaceMesh)
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291

# Generic 3D face model points (approximate, in arbitrary units) corresponding
# to the landmarks above. These don't need to be anatomically exact — they
# just need to be roughly proportionally correct for solvePnP to work.
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),         # Nose tip
    (0.0, -330.0, -65.0),    # Chin
    (-225.0, 170.0, -135.0), # Left eye outer corner
    (225.0, 170.0, -135.0),  # Right eye outer corner
    (-150.0, -150.0, -125.0),# Left mouth corner
    (150.0, -150.0, -125.0), # Right mouth corner
], dtype=np.float64)

# Angle thresholds (degrees) beyond which we consider the head turned away
# rather than facing the camera. Tuned to allow for natural small movement.
YAW_THRESHOLD = 15.0    # left/right turn
PITCH_THRESHOLD = 15.0  # up/down tilt


def _estimate_pose_for_frame(landmarks, frame_width, frame_height):
    """Run solvePnP for a single frame's landmarks.

    Returns (yaw, pitch, roll) in degrees, or None if the solve fails.
    """
    image_points = np.array([
        (landmarks[NOSE_TIP].x * frame_width, landmarks[NOSE_TIP].y * frame_height),
        (landmarks[CHIN].x * frame_width, landmarks[CHIN].y * frame_height),
        (landmarks[LEFT_EYE_OUTER].x * frame_width, landmarks[LEFT_EYE_OUTER].y * frame_height),
        (landmarks[RIGHT_EYE_OUTER].x * frame_width, landmarks[RIGHT_EYE_OUTER].y * frame_height),
        (landmarks[LEFT_MOUTH_CORNER].x * frame_width, landmarks[LEFT_MOUTH_CORNER].y * frame_height),
        (landmarks[RIGHT_MOUTH_CORNER].x * frame_width, landmarks[RIGHT_MOUTH_CORNER].y * frame_height),
    ], dtype=np.float64)

    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Decompose rotation matrix into Euler angles (yaw, pitch, roll)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = 0

    yaw_deg = np.degrees(yaw)
    pitch_deg = np.degrees(pitch)
    roll_deg = np.degrees(roll)

    # Fix Euler angle wraparound: this decomposition can return pitch near
    # +-180 degrees when the physically correct value is near 0. Fold it
    # back into a sane range by reflecting across the wraparound point.
    if pitch_deg > 90:
        pitch_deg -= 180
    elif pitch_deg < -90:
        pitch_deg += 180

    return yaw_deg, pitch_deg, roll_deg


def _classify_direction(yaw, pitch):
    """Classify head direction from yaw/pitch angles."""
    if pitch > PITCH_THRESHOLD:
        return "looking_down"
    elif pitch < -PITCH_THRESHOLD:
        return "looking_up"
    elif yaw > YAW_THRESHOLD:
        return "looking_right"
    elif yaw < -YAW_THRESHOLD:
        return "looking_left"
    else:
        return "facing_camera"


def estimate_head_pose(video_path: str, sample_every_n_frames: int = 3, debug: bool = False) -> dict:
    """Return breakdown of head pose direction percentages over the video.

    Returns a dict like:
    {
        "facing_camera_percent": 78.5,
        "looking_down_percent": 10.2,
        "looking_up_percent": 1.0,
        "looking_left_percent": 5.5,
        "looking_right_percent": 4.8,
        "frames_analyzed": 341
    }
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    direction_counts = {
        "facing_camera": 0,
        "looking_down": 0,
        "looking_up": 0,
        "looking_left": 0,
        "looking_right": 0,
    }
    frames_analyzed = 0
    frame_index = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_index % sample_every_n_frames != 0:
                frame_index += 1
                continue
            current_frame = frame_index
            frame_index += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                continue

            landmarks = results.multi_face_landmarks[0].landmark
            pose = _estimate_pose_for_frame(landmarks, frame_width, frame_height)
            if pose is None:
                continue

            yaw, pitch, roll = pose
            direction = _classify_direction(yaw, pitch)
            direction_counts[direction] += 1
            frames_analyzed += 1

            if debug:
                timestamp = current_frame / fps
                print(f"[{timestamp:5.2f}s] yaw={yaw:6.1f} pitch={pitch:6.1f} -> {direction}")

    cap.release()

    if frames_analyzed == 0:
        return {
            "facing_camera_percent": 0.0,
            "looking_down_percent": 0.0,
            "looking_up_percent": 0.0,
            "looking_left_percent": 0.0,
            "looking_right_percent": 0.0,
            "frames_analyzed": 0,
        }

    return {
        "facing_camera_percent": round(direction_counts["facing_camera"] / frames_analyzed * 100, 1),
        "looking_down_percent": round(direction_counts["looking_down"] / frames_analyzed * 100, 1),
        "looking_up_percent": round(direction_counts["looking_up"] / frames_analyzed * 100, 1),
        "looking_left_percent": round(direction_counts["looking_left"] / frames_analyzed * 100, 1),
        "looking_right_percent": round(direction_counts["looking_right"] / frames_analyzed * 100, 1),
        "frames_analyzed": frames_analyzed,
    }


if __name__ == "__main__":
    result = estimate_head_pose("data/sample_videos/test_clip.mp4", debug=True)
    print("\n--- SUMMARY ---")
    print(result)