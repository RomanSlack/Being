"""Tests for the audio buffer used in real-time streaming."""

import numpy as np
from being.api.server import AudioBuffer


def test_audio_buffer_init():
    buf = AudioBuffer(sample_rate=16000, frame_rate=25)
    assert buf.samples_per_frame == 640


def test_audio_buffer_not_enough_data():
    buf = AudioBuffer(sample_rate=16000, frame_rate=25)
    # Add less than one frame of audio
    chunk = np.zeros(320, dtype=np.int16).tobytes()
    buf.add_chunk(chunk)
    assert buf.get_frame_features() is None


def test_audio_buffer_one_frame():
    buf = AudioBuffer(sample_rate=16000, frame_rate=25)
    # Add exactly one frame of audio (640 samples)
    chunk = np.ones(640, dtype=np.int16).tobytes()
    buf.add_chunk(chunk)
    features = buf.get_frame_features()
    assert features is not None
    assert len(features) == 640


def test_audio_buffer_multiple_frames():
    buf = AudioBuffer(sample_rate=16000, frame_rate=25)
    # Add 3 frames worth of audio
    chunk = np.ones(640 * 3, dtype=np.int16).tobytes()
    buf.add_chunk(chunk)

    # Should be able to get 3 frames
    for _ in range(3):
        features = buf.get_frame_features()
        assert features is not None

    # 4th should return None
    assert buf.get_frame_features() is None


def test_audio_buffer_reset():
    buf = AudioBuffer(sample_rate=16000, frame_rate=25)
    chunk = np.ones(640, dtype=np.int16).tobytes()
    buf.add_chunk(chunk)
    buf.reset()
    assert buf.get_frame_features() is None
