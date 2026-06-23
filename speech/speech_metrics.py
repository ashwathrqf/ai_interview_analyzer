"""Speaking speed (WPM) and pause detection metrics."""


def calculate_wpm(transcript: str, duration_seconds: float) -> float:
    """Return words-per-minute for the given transcript and duration."""
    raise NotImplementedError


def detect_pauses(segments: list) -> dict:
    """Detect pauses between Whisper segments.

    Returns a dict like {"average_pause": 0.74, "pause_count": 12}
    """
    raise NotImplementedError
