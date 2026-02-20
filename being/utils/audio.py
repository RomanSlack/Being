"""Audio feature extraction utilities.

Wraps the various audio feature extractors (wav2vec, HuBERT, DeepSpeech)
with a unified interface for both offline and real-time use.
"""

import numpy as np
from pathlib import Path
from enum import Enum
from typing import Optional


class AudioExtractor(str, Enum):
    DEEPSPEECH = "deepspeech"
    WAV2VEC = "wav2vec"
    HUBERT = "hubert"
    AVE = "ave"


class AudioFeatureExtractor:
    """Unified audio feature extraction.

    Loads the specified model once and provides both file-based
    and streaming feature extraction.
    """

    def __init__(self, extractor: AudioExtractor = AudioExtractor.WAV2VEC, device: str = "cuda:0"):
        self.extractor = extractor
        self.device = device
        self._model = None
        self._processor = None

    def load(self):
        """Load the feature extraction model into memory."""
        if self.extractor == AudioExtractor.WAV2VEC:
            self._load_wav2vec()
        elif self.extractor == AudioExtractor.HUBERT:
            self._load_hubert()
        elif self.extractor == AudioExtractor.DEEPSPEECH:
            self._load_deepspeech()
        elif self.extractor == AudioExtractor.AVE:
            pass  # AVE is handled differently (at training time)

    def extract_from_file(self, audio_path: Path) -> np.ndarray:
        """Extract features from an audio file.

        Returns:
            np.ndarray of shape (num_frames, feature_dim)
        """
        import soundfile as sf
        audio, sr = sf.read(str(audio_path))

        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        return self.extract_from_array(audio, sr)

    def extract_from_array(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract features from a numpy audio array.

        Args:
            audio: Audio samples, float32 in [-1, 1].
            sample_rate: Sample rate (should be 16000).

        Returns:
            np.ndarray of shape (num_frames, feature_dim)
        """
        if self.extractor == AudioExtractor.WAV2VEC:
            return self._extract_wav2vec(audio, sample_rate)
        elif self.extractor == AudioExtractor.HUBERT:
            return self._extract_hubert(audio, sample_rate)
        elif self.extractor == AudioExtractor.DEEPSPEECH:
            return self._extract_deepspeech(audio, sample_rate)
        else:
            raise ValueError(f"Unsupported extractor for real-time: {self.extractor}")

    def extract_chunk(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """Extract features from a streaming audio chunk.

        For real-time use — maintains internal state for windowed extraction.
        Returns None if not enough audio has been accumulated.
        """
        # TODO: Implement streaming feature extraction with proper windowing
        # This requires maintaining a sliding window buffer
        return self.extract_from_array(audio_chunk, sample_rate)

    def _load_wav2vec(self):
        """Load wav2vec 2.0 model."""
        import torch
        from transformers import Wav2Vec2Processor, Wav2Vec2Model

        self._processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self._model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
        self._model.eval()

    def _load_hubert(self):
        """Load HuBERT model."""
        import torch
        from transformers import HubertModel, Wav2Vec2Processor

        self._processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        self._model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to(self.device)
        self._model.eval()

    def _load_deepspeech(self):
        """Load DeepSpeech model."""
        # DeepSpeech uses a different extraction pipeline
        # (python_speech_features → MLP). We delegate to InsTaG's implementation.
        pass

    def _extract_wav2vec(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract wav2vec 2.0 features."""
        import torch

        inputs = self._processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self._model(inputs.input_values.to(self.device))

        # outputs.last_hidden_state: (1, seq_len, 768)
        features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        return features

    def _extract_hubert(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract HuBERT features."""
        import torch

        inputs = self._processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self._model(inputs.input_values.to(self.device))

        features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        return features

    def _extract_deepspeech(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract DeepSpeech features using python_speech_features."""
        from python_speech_features import mfcc

        # Standard DeepSpeech feature extraction: 29-dim MFCC
        features = mfcc(audio, samplerate=sample_rate, numcep=29)
        return features

    def unload(self):
        """Free model memory."""
        import torch
        self._model = None
        self._processor = None
        torch.cuda.empty_cache()
