#!/usr/bin/env python3
"""
Simple WAV -> text tester for Vosk.

Usage:
  python tools/transcribe_wav_vosk.py --model /path/to/vosk-model --wav test.wav

This prints incremental results and the final transcript.
"""
import argparse
import json
import wave
import sys
import numpy as np

try:
    from vosk import Model, KaldiRecognizer
except Exception as e:
    sys.exit("vosk not installed. Install with: pip install vosk")


def transcribe(wav_path: str, model_path: str, chunk_frames: int = 4000):
    wf = wave.open(wav_path, "rb")
    nchannels = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    rate = wf.getframerate()

    model = Model(model_path)
    rec = KaldiRecognizer(model, rate)
    print(f"Transcribing '{wav_path}' @ {rate} Hz using model '{model_path}' (channels={nchannels}, sampwidth={sampwidth})")

    def _downmix_to_mono_bytes(frame_bytes: bytes, channels: int, width: int) -> bytes:
        # Currently only supports 16-bit PCM (width==2). For other widths, return raw bytes.
        if width != 2 or channels == 1:
            return frame_bytes
        arr = np.frombuffer(frame_bytes, dtype=np.int16)
        try:
            arr = arr.reshape(-1, channels)
        except Exception:
            return frame_bytes
        mono = arr.mean(axis=1).astype(np.int16)
        return mono.tobytes()

    while True:
        data = wf.readframes(chunk_frames)
        if len(data) == 0:
            break
        if nchannels > 1 and sampwidth == 2:
            data_mono = _downmix_to_mono_bytes(data, nchannels, sampwidth)
        else:
            data_mono = data

        if rec.AcceptWaveform(data_mono):
            res = json.loads(rec.Result())
            text = res.get("text", "")
            if text:
                print("Result:", text)
        else:
            pr = json.loads(rec.PartialResult())
            partial = pr.get("partial", "")
            if partial:
                print("Partial:", partial, end="\r")

    final = json.loads(rec.FinalResult()).get("text", "")
    print("\nFinal:", final)


def main():
    p = argparse.ArgumentParser(description="Transcribe WAV with Vosk")
    p.add_argument("--model", required=True, help="Path to Vosk model directory")
    p.add_argument("--wav", default="test.wav", help="Path to WAV file (mono, 16-bit/PCM)")
    args = p.parse_args()
    transcribe(args.wav, args.model)


if __name__ == "__main__":
    main()
