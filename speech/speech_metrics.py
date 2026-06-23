"""Speaking speed (WPM) and pause detection metrics."""


def calculate_wpm(transcript: str, duration_seconds: float) -> float:
    """Return words-per-minute for the given transcript and duration."""
    word_count = len(transcript.split())
    minutes = duration_seconds / 60
    if minutes == 0:
        return 0.0
    return round(word_count / minutes, 1)


def detect_pauses(segments: list, min_pause_seconds: float = 0.3) -> dict:
    """Detect pauses between Whisper segments.

    A "pause" is the gap between when one segment ends and the next begins.
    Gaps shorter than min_pause_seconds are treated as natural speech flow,
    not a pause, to avoid counting tiny Whisper segmentation artifacts.

    Args:
        segments: list of Whisper segment dicts, each with "start" and "end"
        min_pause_seconds: minimum gap length to count as a real pause

    Returns:
        dict like {"average_pause": 0.74, "pause_count": 12, "pauses": [...]}
    """
    pauses = []

    for i in range(len(segments) - 1):
        current_end = segments[i]["end"]
        next_start = segments[i + 1]["start"]
        gap = next_start - current_end

        if gap >= min_pause_seconds:
            pauses.append(round(gap, 2))

    if not pauses:
        return {"average_pause": 0.0, "pause_count": 0, "pauses": []}

    average_pause = round(sum(pauses) / len(pauses), 2)
    return {
        "average_pause": average_pause,
        "pause_count": len(pauses),
        "pauses": pauses,
    }


def wpm_feedback(wpm: float, ideal_min: int = 120, ideal_max: int = 160) -> str:
    """Return a human-readable label for a WPM value."""
    if wpm < ideal_min:
        return "Slow — consider picking up the pace slightly"
    elif wpm > ideal_max:
        return "Fast — consider slowing down for clarity"
    else:
        return "Excellent — within the ideal range"


if __name__ == "__main__":
    # Quick manual test using real values from your medium-model transcript.
    sample_transcript = (
        "Hi, I'm Ashwath. I'm studying Mechanical Engineering in IIT Madras. "
        "I live in Bangalore, my favorite hobbies are reading books, going "
        "through YouTube stuff, any new challenges and creating projects "
        "like this. I love doing stuff like this. I want to get into "
        "quantitative finance in NFO or machine learning and data analytics. "
        "And my priority of the night I may say is my honesty on my talent "
        "to lead a team and to work under pressure. Thank you."
    )
    sample_segments = [
        {"start": 0.00, "end": 10.40},
        {"start": 10.40, "end": 20.80},
        {"start": 20.80, "end": 25.36},
        {"start": 28.24, "end": 34.80},
        {"start": 35.44, "end": 46.80},
        {"start": 46.80, "end": 50.80},
    ]

    duration = sample_segments[-1]["end"] - sample_segments[0]["start"]
    wpm = calculate_wpm(sample_transcript, duration)
    print(f"WPM: {wpm} -> {wpm_feedback(wpm)}")

    pause_info = detect_pauses(sample_segments)
    print(f"Pause info: {pause_info}")