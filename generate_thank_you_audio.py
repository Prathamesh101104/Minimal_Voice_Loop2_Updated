import os
import io
import wave
import base64
import requests
import numpy as np
import soxr
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SARVAM_KEY = os.getenv("SARVAM_API_KEY")
MODEL = os.getenv("SARVAM_MODEL", "bulbul:v3")
VOICE = os.getenv("SARVAM_VOICE", "priya")
LANGUAGE = os.getenv("SARVAM_LANGUAGE", "en-IN")

THANK_YOU_TEXT = "Thank you for calling Nimbus. Have a wonderful day! Goodbye."

print(f"Synthesizing thank-you audio using Sarvam ({MODEL}, voice: {VOICE})...")

resp = requests.post(
    "https://api.sarvam.ai/text-to-speech",
    headers={"api-subscription-key": SARVAM_KEY},
    json={
        "inputs": [THANK_YOU_TEXT],
        "target_language_code": LANGUAGE,
        "speaker": VOICE,
        "model": MODEL,
    },
    timeout=30,
)

if resp.status_code != 200:
    print(f"Error: {resp.status_code} - {resp.text}")
    exit(1)

data = resp.json()
audio_base64 = data["audios"][0]
audio_bytes = base64.b64decode(audio_base64)

# Sarvam returns a full WAV container
with wave.open(io.BytesIO(audio_bytes), "rb") as w_in:
    in_sr = w_in.getframerate()
    in_channels = w_in.getnchannels()
    in_sampwidth = w_in.getsampwidth()
    in_frames = w_in.readframes(w_in.getnframes())

print(f"Received audio from Sarvam: {in_sr}Hz, {in_channels} channels, {len(in_frames)} bytes")

assets_dir = Path("assets")
assets_dir.mkdir(parents=True, exist_ok=True)

# Save original 22050Hz
with wave.open(str(assets_dir / "thank_you.wav"), "wb") as w_out:
    w_out.setnchannels(in_channels)
    w_out.setsampwidth(in_sampwidth)
    w_out.setframerate(in_sr)
    w_out.writeframes(in_frames)

# Resample to 8000Hz mono PCM for Vobiz telephony
audio_array = np.frombuffer(in_frames, dtype=np.int16)
resampled_array = soxr.resample(audio_array, in_sr, 8000)
resampled_int16 = np.clip(resampled_array, -32768, 32767).astype(np.int16)
raw_8k = resampled_int16.tobytes()

with wave.open(str(assets_dir / "thank_you_8k.wav"), "wb") as w_out:
    w_out.setnchannels(1)
    w_out.setsampwidth(2)
    w_out.setframerate(8000)
    w_out.writeframes(raw_8k)

print(f"Saved: assets/thank_you_8k.wav ({len(raw_8k)} bytes, 8000Hz, ~{len(raw_8k)/16000:.2f} seconds)")
