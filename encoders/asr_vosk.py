"""
Simple Vosk ASR wrapper for demo transcripts

Usage:
    asr = VoskASR(model_path='/path/to/vosk-model-small-en-us-0.15', sample_rate=16000)
    transcript = asr.transcribe(audio_tensor)

`transcribe` expects a CPU torch.Tensor or numpy array with shape (batch, samples)
or (samples,) and returns a short transcript string.
"""
import os
import json
import numpy as np
import torch
import urllib.request
import tempfile
import shutil
import zipfile
import tarfile

try:
    from vosk import Model, KaldiRecognizer
except Exception:
    Model = None
    KaldiRecognizer = None


class VoskASR:
    def __init__(self, model_path: str = None, model_name: str = 'vosk-model-small-en-us-0.15', sample_rate: int = 16000, download_dir: str = None):
        if Model is None:
            raise ImportError("vosk is not installed. Install with 'pip install vosk'.")

        self.sample_rate = sample_rate
        # Determine model path: if provided use it, otherwise download by model_name
        if model_path and os.path.exists(model_path):
            chosen_path = model_path
        else:
            # Prepare download directory
            base_dir = download_dir or os.path.join(os.getcwd(), 'models', 'vosk')
            os.makedirs(base_dir, exist_ok=True)
            chosen_path = os.path.join(base_dir, model_name)
            if not os.path.exists(chosen_path):
                # Attempt to download and extract
                try:
                    self._download_and_extract_model(model_name, base_dir)
                except Exception as e:
                    raise RuntimeError(f"Failed to download/extract Vosk model '{model_name}': {e}")

        if not os.path.exists(chosen_path):
            raise FileNotFoundError(f"Vosk model not found at {chosen_path}")

        self.model = Model(chosen_path)

    def _to_pcm_bytes(self, arr: np.ndarray) -> bytes:
        # Expect float32 in range roughly [-1,1] or int16 already
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            # clip then convert
            clipped = np.clip(arr, -1.0, 1.0)
            ints = (clipped * 32767).astype(np.int16)
        elif arr.dtype == np.int16:
            ints = arr
        else:
            ints = arr.astype(np.int16)
        return ints.tobytes()

    def transcribe(self, audio_tensor) -> str:
        """Transcribe one batch of audio and return a short transcript string."""
        # Accept torch.Tensor or numpy array
        if isinstance(audio_tensor, torch.Tensor):
            audio_np = audio_tensor.detach().cpu().numpy()
        else:
            audio_np = np.asarray(audio_tensor)

        # If batch dimension present, take first example
        if audio_np.ndim == 2:
            audio_np = audio_np[0]

        # Ensure mono 1D
        if audio_np.ndim != 1:
            audio_np = audio_np.flatten()

        pcm = self._to_pcm_bytes(audio_np)

        rec = KaldiRecognizer(self.model, self.sample_rate)
        rec.AcceptWaveform(pcm)
        res = rec.Result()
        try:
            j = json.loads(res)
            return j.get('text', '')
        except Exception:
            return ''

    def _download_and_extract_model(self, model_name: str, dest_dir: str):
        """Download and extract a Vosk model into dest_dir/model_name.

        This method downloads from the official Vosk model hosting site.
        """
        # Try common archive extensions
        urls = [
            f"https://alphacephei.com/vosk/models/{model_name}.zip",
            f"https://alphacephei.com/vosk/models/{model_name}.tar.gz",
        ]

        tmp_dir = tempfile.mkdtemp()
        try:
            for url in urls:
                try:
                    fname = os.path.join(tmp_dir, os.path.basename(url))
                    urllib.request.urlretrieve(url, fname)
                    # Extract
                    if fname.endswith('.zip'):
                        with zipfile.ZipFile(fname, 'r') as zf:
                            zf.extractall(dest_dir)
                    elif fname.endswith('.tar.gz') or fname.endswith('.tgz'):
                        with tarfile.open(fname, 'r:gz') as tf:
                            tf.extractall(dest_dir)
                    else:
                        # Unknown format, skip
                        continue

                    # After extraction, expect a folder named model_name inside dest_dir
                    extracted_path = os.path.join(dest_dir, model_name)
                    if os.path.exists(extracted_path):
                        return
                except Exception:
                    # Try next URL
                    continue

            raise RuntimeError(f"Unable to download or extract model {model_name}. Tried URLs: {urls}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class StreamingVosk:
    """Persistent streaming Vosk wrapper.

    Use `feed(audio_tensor)` repeatedly with short chunks (e.g. 200-500ms)
    to get partial and final transcript updates.
    """
    def __init__(self, model_path: str = None, model_name: str = 'vosk-model-small-en-us-0.15', sample_rate: int = 16000, download_dir: str = None):
        if Model is None:
            raise ImportError("vosk is not installed. Install with 'pip install vosk'.")

        self.sample_rate = sample_rate
        if model_path and os.path.exists(model_path):
            chosen_path = model_path
        else:
            base_dir = download_dir or os.path.join(os.getcwd(), 'models', 'vosk')
            os.makedirs(base_dir, exist_ok=True)
            chosen_path = os.path.join(base_dir, model_name)
            if not os.path.exists(chosen_path):
                try:
                    # reuse download logic from VoskASR
                    VoskASR(model_path=model_path, model_name=model_name, sample_rate=sample_rate, download_dir=download_dir)
                except Exception as e:
                    raise RuntimeError(f"Failed to download/extract Vosk model '{model_name}': {e}")

        if not os.path.exists(chosen_path):
            raise FileNotFoundError(f"Vosk model not found at {chosen_path}")

        self.model = Model(chosen_path)
        self.rec = KaldiRecognizer(self.model, self.sample_rate)

    def _to_pcm_bytes(self, arr: np.ndarray) -> bytes:
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            clipped = np.clip(arr, -1.0, 1.0)
            ints = (clipped * 32767).astype(np.int16)
        elif arr.dtype == np.int16:
            ints = arr
        else:
            ints = arr.astype(np.int16)
        return ints.tobytes()

    def feed(self, audio_tensor) -> str:
        """Feed a short audio chunk (torch.Tensor or numpy array) and return transcript string.

        Returns the partial transcript if available, otherwise final text when endpointing occurs.
        """
        if isinstance(audio_tensor, torch.Tensor):
            audio_np = audio_tensor.detach().cpu().numpy()
        else:
            audio_np = np.asarray(audio_tensor)

        if audio_np.ndim == 2:
            audio_np = audio_np[0]
        if audio_np.ndim != 1:
            audio_np = audio_np.flatten()

        pcm = self._to_pcm_bytes(audio_np)
        try:
            accepted = self.rec.AcceptWaveform(pcm)
        except Exception:
            # If recognizer errors, return empty
            return ""

        if accepted:
            try:
                res = json.loads(self.rec.Result())
                return res.get('text', '')
            except Exception:
                return ""
        else:
            try:
                pr = json.loads(self.rec.PartialResult())
                return pr.get('partial', '')
            except Exception:
                return ""

    def reset(self):
        self.rec = KaldiRecognizer(self.model, self.sample_rate)

    def close(self):
        # nothing special to free
        pass
