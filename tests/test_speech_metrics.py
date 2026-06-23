"""Tests for speech metrics: WPM calculation and pause detection."""

from speech.speech_metrics import calculate_wpm, detect_pauses, wpm_feedback
from speech.filler_words import count_filler_words


def test_calculate_wpm_matches_known_value():
    # Real transcript + duration from project testing: ~85 words over ~52s
    transcript = " ".join(["word"] * 85)
    wpm = calculate_wpm(transcript, 52.0)
    assert wpm == 98.1


def test_calculate_wpm_zero_duration_returns_zero():
    assert calculate_wpm("some words here", 0) == 0.0


def test_detect_pauses_finds_real_gap():
    # Mirrors the actual gap we found in test_clip.mp4 between segments
    segments = [
        {"start": 0.00, "end": 25.36},
        {"start": 28.24, "end": 50.80},
    ]
    result = detect_pauses(segments)
    assert result["pause_count"] == 1
    assert result["average_pause"] == 2.88


def test_detect_pauses_ignores_tiny_gaps():
    # Gaps under min_pause_seconds (0.3s default) shouldn't count as pauses
    segments = [
        {"start": 0.00, "end": 10.00},
        {"start": 10.10, "end": 20.00},  # 0.1s gap, too small to count
    ]
    result = detect_pauses(segments)
    assert result["pause_count"] == 0


def test_wpm_feedback_labels():
    assert "Slow" in wpm_feedback(90)
    assert "Excellent" in wpm_feedback(140)
    assert "Fast" in wpm_feedback(200)


def test_count_filler_words_matches_known_transcript():
    # Real transcript from project testing — known to contain exactly 2
    # instances of "like" and no other filler words
    transcript = (
        "Hi, I'm Ashwath. I'm studying Mechanical Engineering in IIT Madras. "
        "I live in Bangalore, my favorite hobbies are reading books, going "
        "through YouTube stuff, any new challenges and creating projects "
        "like this. I love doing stuff like this. I want to get into "
        "quantitative finance in NFO or machine learning and data analytics. "
        "And my priority of the night I may say is my honesty on my talent "
        "to lead a team and to work under pressure. Thank you."
    )
    result = count_filler_words(transcript)
    assert result["like"] == 2
    assert result["total"] == 2
    assert result["uh"] == 0