"""Basic tests for speech metrics."""

import pytest

from speech.speech_metrics import calculate_wpm


def test_calculate_wpm_placeholder():
    with pytest.raises(NotImplementedError):
        calculate_wpm("hello world this is a test", 10)
