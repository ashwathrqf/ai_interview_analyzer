"""Smile frequency detection using MediaPipe FaceMesh.

Approach: smiling pulls the mouth corners outward (wider) and slightly
upward relative to the lips' resting position. We compute a ratio of
mouth width to face width (which increases when smiling, since the face
width itself doesn't change but the mouth stretches), combined with how
far the mouth corners sit above the midpoint of the lips (corners lift
when smiling, droop or stay neutral otherwise).

This combined "smile score" is then compared against a threshold to
classify each frame as smiling or neutral.
"""

import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Landmark indices (MediaPipe FaceMesh)
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291
UPPER_LIP_TOP = 13
LOWER_LIP_BOTTOM = 14
FACE_LEFT = 234   # left cheek/face edge, used to normalize against face width
FACE_RIGHT = 454  # right cheek/face edge

# A smile score above this threshold counts as "smiling" for that frame.
# Tuned conservatively; see calibration notes if results look off.
SMILE_THRESHOLD = 0.55


def _smile_score(landmarks):
    """Return a normalized smile score for one frame.

    Combines mouth-width-to-face-width ratio with corner lift relative to
    lip center. Higher score = more smile-like.
    """
    left_corner = landmarks[MOUTH_LEFT_CORNER]
    right_corner = landmarks[MOUTH_RIGHT_CORNER]
    upper_lip = landmarks[UPPER_LIP_TOP]
    lower_lip = landmarks[LOWER_LIP_BOTTOM]
    face_left = landmarks[FACE_LEFT]
    face_right = landmarks[FACE_RIGHT]

    mouth_width = abs(right_corner.x - left_corner.x)
    face_width = abs(face_right.x - face_left.x)
    if face_width == 0:
        return 0.0
    width_ratio = mouth_width / face_width

    lip_center_y = (upper_lip.y + lower_lip.y) / 2
    corner_avg_y = (left_corner.y + right_corner.y) / 2

    # Normalize corner lift against face width instead of lip gap height.
    # Lip gap height collapses toward zero whenever the mouth is closed
    # (e.g. mid-speech, between words), which previously caused this ratio
    # to explode toward huge, meaningless values. Face width is stable
    # regardless of mouth state, so it's a safe normalizer here.
    corner_lift_raw = lip_center_y - corner_avg_y  # positive when corners lifted
    corner_lift = corner_lift_raw / face_width

    # Combine: width stretch matters most, corner lift adds confirmation.
    # Clip corner_lift's contribution so a single noisy frame can't dominate.
    score = (width_ratio * 0.7) + (max(min(corner_lift, 0.1), 0) * 3.0)
    return score


def detect_smile_frequency(video_path: str, sample_every_n_frames: int = 3, debug: bool = False) -> dict:
    """Return percentage of analyzed frames where the subject is smiling.

    Returns:
        dict like {"smile_percent": 21.4, "frames_analyzed": 341}
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    smiling_count = 0
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

            frames_analyzed += 1
            landmarks = results.multi_face_landmarks[0].landmark
            score = _smile_score(landmarks)
            is_smiling = score >= SMILE_THRESHOLD

            if debug:
                timestamp = current_frame / fps
                marker = "SMILE" if is_smiling else "neutral"
                print(f"[{timestamp:5.2f}s] score={score:.3f} -> {marker}")

            if is_smiling:
                smiling_count += 1

    cap.release()

    if frames_analyzed == 0:
        return {"smile_percent": 0.0, "frames_analyzed": 0}

    smile_percent = round((smiling_count / frames_analyzed) * 100, 1)
    return {"smile_percent": smile_percent, "frames_analyzed": frames_analyzed}


if __name__ == "__main__":
    result = detect_smile_frequency("data/sample_videos/test_clip.mp4", debug=True)
    print("\n--- SUMMARY ---")
    print(result)