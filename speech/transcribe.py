"""Speech-to-text using Whisper (runs locally, free)."""

import whisper

# Loading the model is slow (a few seconds), so we cache it instead of
# reloading on every call. "base" is a good accuracy/speed tradeoff for
# short clips on a laptop; "tiny" is faster but less accurate, "small" is
# slower but more accurate.
_model = None
MODEL_SIZE = "medium"


def _get_model():
    global _model
    if _model is None:
        _model = whisper.load_model(MODEL_SIZE)
    return _model


def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio_path using Whisper.

    Returns a dict with at least: {"text": str, "segments": list}
    Each segment has "start", "end", and "text" — useful later for pause
    detection in speech_metrics.py.
    """
    model = _get_model()
    result = model.transcribe(audio_path)
    return result


if __name__ == "__main__":
    # Quick manual test against the audio we extracted in the previous step.
    result = transcribe_audio("data/sample_videos/test_clip_audio.wav")
    print("--- TRANSCRIPT ---")
    print(result["text"])
    print("\n--- SEGMENTS ---")
    for seg in result["segments"]:
        print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")