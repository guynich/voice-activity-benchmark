#
# Copyright 2021-2025 Picovoice Inc.
#
# You may not use this file except in compliance with the license.
# A copy of the license is located in the "LICENSE" file accompanying this source.
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#

import os
import sys
from collections import namedtuple
from enum import Enum

import numpy as np
import pvcobra

from mixer import DEFAULT_SAMPLERATE


ThresholdInfo = namedtuple('ThresholdInfo', 'min, max, step')

class Engines(Enum):
    COBRA = 'Cobra'
    SILERO = 'Silero'
    TENVAD = 'TEN-VAD'
    WEBRTC = 'WebRTC'
    WEBRTCRNNCLIDUMMY = 'WebRTCRNNCLIDummy'

engine_create_map = {
    Engines.COBRA: lambda threshold, access_key, **kwargs: CobraEngine(threshold, access_key),
    Engines.SILERO: lambda threshold, **kwargs: SileroEngine(threshold),
    Engines.TENVAD: lambda threshold, **kwargs: TENVADEngine(threshold),
    Engines.WEBRTC: lambda threshold, **kwargs: WebRTCEngine(threshold),
    Engines.WEBRTCRNNCLIDUMMY: lambda threshold, **kwargs: WebRTCRNNCLIDummyEngine(threshold),
}

threshold_info_map = {
    Engines.COBRA: ThresholdInfo(0.0, 1.0, 0.001),
    Engines.SILERO: ThresholdInfo(0.0, 1.0, 0.001),
    Engines.TENVAD: ThresholdInfo(0.0, 1.0, 0.01),  # Different threshold spacing to Cobra/Silero.
    Engines.WEBRTC: ThresholdInfo(0, 3, 1),
    Engines.WEBRTCRNNCLIDUMMY: ThresholdInfo(0.0, 1.0, 0.01),
}


class Engine(object):
    def process(self, pcm, frame_key):
        raise NotImplementedError()

    def frame_length(self):
        raise NotImplementedError()

    def release(self):
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__

    @staticmethod
    def threshold_info(engine_type):
        if engine_type in threshold_info_map:
            return threshold_info_map[engine_type]
        else:
            raise ValueError("no threshold range for '%s'", engine_type.value)

    @staticmethod
    def create(engine, threshold, **kwargs):
        if engine in engine_create_map:
            return engine_create_map[engine](threshold, **kwargs)
        else:
            raise ValueError("cannot create engine of type '%s'", engine.value)


class CobraEngine(Engine):
    _cache = dict()

    def __init__(self, threshold, access_key):
        self._cobra = pvcobra.create(access_key=access_key)
        self._threshold = threshold

    def process(self, pcm, frame_key):
        assert pcm.dtype == np.int16

        if frame_key in self._cache:
            voice_probability = self._cache[frame_key]
        else:
            voice_probability = self._cobra.process(pcm)
            self._cache[frame_key] = voice_probability

        return voice_probability >= self._threshold

    def frame_length(self):
        return self._cobra.frame_length

    def release(self):
        self._cobra.delete()


class SileroEngine(Engine):
    _cache = dict()

    def __init__(self, threshold):
        from silero_vad import load_silero_vad
        self._model = load_silero_vad(onnx=True)
        self._threshold = threshold
        import torch
        self._torch = torch

    def process(self, pcm, frame_key):
        assert pcm.dtype == np.int16

        if frame_key in self._cache:
            voice_probability = self._cache[frame_key]
        else:
            pcm_float = pcm.astype(np.float32) / np.iinfo(np.int16).max
            voice_probability = self._model(self._torch.from_numpy(pcm_float), DEFAULT_SAMPLERATE).item()
            self._cache[frame_key] = voice_probability

        return (voice_probability >= self._threshold)

    def frame_length(self):
        assert DEFAULT_SAMPLERATE in (16000, 8000)
        return 512 if DEFAULT_SAMPLERATE == 16000 else 256

    def release(self):
        del self._model


class TENVADEngine(Engine):
    _cache = dict()

    def __init__(self, threshold):
        # Add lib directory to Python path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))

        import ten_vad_python

        self._model = ten_vad_python.VAD(
            hop_size=self.frame_length(), threshold=threshold
        )
        self._threshold = threshold

    def process(self, pcm, frame_key):
        assert pcm.dtype == np.int16

        if frame_key in self._cache:
            voice_probability = self._cache[frame_key]
        else:
            # Ensure frame is exactly hop_size
            if len(pcm) < self.frame_length():
                # Pad with zeros if needed
                pcm = np.pad(
                    pcm,
                    (0, self.frame_length() - len(pcm)),
                    mode="constant",
                )
            voice_probability, _ = self._model.process(pcm)
            self._cache[frame_key] = voice_probability
        # TODO(guy): return is_voice, not (voice_probability >= self._threshold).
        return voice_probability >= self._threshold

    def frame_length(self):
        assert DEFAULT_SAMPLERATE == 16000
        return 256

    def release(self):
        del self._model


class WebRTCEngine(Engine):
    _FRAME_SEC = 0.03

    def __init__(self, threshold):
        import webrtcvad
        self._vad = webrtcvad.Vad(int(threshold))

    def process(self, pcm, frame_key):
        assert pcm.dtype == np.int16

        return self._vad.is_speech(pcm.tobytes(), DEFAULT_SAMPLERATE)

    def frame_length(self):
        return int((DEFAULT_SAMPLERATE * self._FRAME_SEC))

    def release(self):
        del self._vad


class WebRTCRNNCLIDummyEngine(Engine):
    """ Dummy implementation of WebRTCRNNVAD that reads precomputed probabilities from a file generated from the CLI: daanzu/webrtc_rnnvad: webrtc_rnnvad (https://github.com/daanzu/webrtc_rnnvad) """

    def __init__(self, threshold):
        self._threshold = threshold
        file_name = 'webrtcrnnvad_probs.bin'
        # fwrite(&vad_probability, sizeof(float), 1, vad_probs_file);
        with open(file_name, 'rb') as f:
            self._voice_probabilities = np.frombuffer(f.read(), dtype=np.float32)
        self._voice_probabilities_index = 0

    def process(self, pcm, frame_key):
        assert pcm.dtype == np.int16

        voice_probability = self._voice_probabilities[self._voice_probabilities_index]
        self._voice_probabilities_index += 1

        return (voice_probability >= self._threshold)

    def frame_length(self):
        return 160

    def release(self):
        assert self._voice_probabilities_index == len(self._voice_probabilities)
        del self._voice_probabilities
