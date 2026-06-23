"""Eye contact estimation using MediaPipe FaceMesh.

Approach: "Looking at the camera" requires two things to both be true:
  1. The head is roughly facing the camera (not turned left/right/down/up)
  2. The iris is roughly centered within the eye socket (not just eyes
     darting to the side while the head stays still)

Checking iris position alone is insufficient: a head turn keeps the iris
roughly centered relative to the (also turned) eye corners, so head pose
must be checked too. This module reuses the head pose logic from
head_pose.py rather than duplicating it.
"""

import cv2
import mediapipe as mp

from vision.head_pose import _estimate_pose_for_frame, YAW_THRESHOLD, PITCH_THRESHOLD

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_CORNERS = (33, 133)
LEFT_IRIS_CENTER = 468
RIGHT_EYE_CORNERS = (362, 263)
RIGHT_IRIS_CENTER = 473

GAZE_THRESHOLD = 0.35


def _iris_offset_ratio(landmarks, iris_idx, corner_a_idx, corner_b_idx):
    iris = landmarks[iris_idx]
    corner_a = landmarks[corner_a_idx]
    corner_b = landmarks[corner_b_idx]

    eye_width = abs(corner_b.x - corner_a.x)
    if eye_width == 0:
        return 0.0

    eye_midpoint_x = (corner_a.x + corner_b.x) / 2
    offset = abs(iris.x - eye_midpoint_x)
    return offset / eye_width


def estimate_eye_contact(video_path: str, sample_every_n_frames: int = 3, debug: bool = False) -> dict:
    """Return eye contact stats over the video.

    A frame counts as "looking at camera" only if the head is facing the
    camera (within YAW_THRESHOLD/PITCH_THRESHOLD) AND the iris is centered
    (within GAZE_THRESHOLD).

    Returns:
        dict like {"eye_contact_percent": 68.3, "frames_analyzed": 341,
                    "frames_with_face_detected": 341}
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    looking_at_camera_count = 0
    frames_with_face = 0
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

            frames_with_face += 1
            landmarks = results.multi_face_landmarks[0].landmark

            # Gate 1: head pose must be roughly facing the camera
            pose = _estimate_pose_for_frame(landmarks, frame_width, frame_height)
            head_facing_camera = False
            if pose is not None:
                yaw, pitch, _ = pose
                head_facing_camera = (
                    abs(yaw) <= YAW_THRESHOLD and abs(pitch) <= PITCH_THRESHOLD
                )

            # Gate 2: iris must be roughly centered
            left_offset = _iris_offset_ratio(landmarks, LEFT_IRIS_CENTER, *LEFT_EYE_CORNERS)
            right_offset = _iris_offset_ratio(landmarks, RIGHT_IRIS_CENTER, *RIGHT_EYE_CORNERS)
            avg_offset = (left_offset + right_offset) / 2
            iris_centered = avg_offset <= GAZE_THRESHOLD

            is_eye_contact = head_facing_camera and iris_centered

            if debug:
                timestamp = current_frame / fps
                marker = "CONTACT" if is_eye_contact else "AWAY"
                print(
                    f"[{timestamp:5.2f}s] head_facing={head_facing_camera} "
                    f"iris_offset={avg_offset:.3f} -> {marker}"
                )

            if is_eye_contact:
                looking_at_camera_count += 1

    cap.release()

    if frames_with_face == 0:
        return {
            "eye_contact_percent": 0.0,
            "frames_analyzed": 0,
            "frames_with_face_detected": 0,
        }

    eye_contact_percent = round((looking_at_camera_count / frames_with_face) * 100, 1)
    return {
        "eye_contact_percent": eye_contact_percent,
        "frames_analyzed": frames_with_face,
        "frames_with_face_detected": frames_with_face,
    }


if __name__ == "__main__":
    result = estimate_eye_contact("data/sample_videos/test_clip.mp4", debug=True)
    print("\n--- SUMMARY ---")
    print(result)