"""Extract audio track from an uploaded interview video."""

from moviepy import VideoFileClip


def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio from video_path and save as a wav file at output_path.

    Args:
        video_path: path to the input video (e.g. .mp4, .mov)
        output_path: path to save the extracted audio (e.g. .wav)

    Returns:
        output_path on success.
    """
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_path)
    clip.close()
    return output_path


if __name__ == "__main__":
    # Quick manual test: run this file directly to confirm extraction works
    # on your sample clip before wiring it into the rest of the app.
    result = extract_audio(
        "data/sample_videos/test_clip.mp4",
        "data/sample_videos/test_clip_audio.wav",
    )
    print(f"Audio extracted to: {result}")