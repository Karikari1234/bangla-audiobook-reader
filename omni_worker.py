"""Bangla TTS worker: runs OmniVoice-MLX (GPU) in its own venv, spawned by library_server.py.

stdin: one JSON line {"text": ...} per paragraph. stdout: one JSON line {"words": [...], "audio": <base64 wav>}.
Settings measured on an M2: 16 steps + prefix cache refresh 8 + uncond reuse 4 = RTF ~0.4.
"""
import base64, io, json, os, re, sys

out = os.fdopen(os.dup(1), "w")  # keep the real stdout for results; library logs go to stderr
os.dup2(2, 1)
import numpy as np, soundfile as sf
from omnivoice.mlx.omnivoice import OmniVoiceMLX

MODEL = os.environ.get("OMNIVOICE_MODEL", "mlx-community/OmniVoice-8bit")  # local dir or Hugging Face id
REF_WAV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "voice_bn.wav")  # synthetic voice made with OmniVoice voice design
REF_TEXT = "গুরুচরণের ছুই চোখ অশ্রুপূর্ণ হইয়৷ উঠিল। শেখর চুপ করিয়া রহিল ।"  # what voice_bn.wav says

m = OmniVoiceMLX.from_pretrained(MODEL, audio_tokenizer_device="mps")
voice = m.create_voice_clone_prompt(REF_WAV, ref_text=REF_TEXT)
SR = m.sampling_rate

for line in sys.stdin:
    text = json.loads(line)["text"]
    audio, words, offset = [], [], 0.0
    for s in [s for s in re.split(r"(?<=[।?!])\s+", text) if s.strip()]:
        a = m.generate(text=s, language="Bengali", voice_clone_prompt=voice, num_step=16, prefix_refresh=8, uncond_reuse=4)[0]
        dur = len(a) / SR
        # no word timestamps from OmniVoice: spread the sentence's duration over its words by length
        ws = s.split()
        total = sum(map(len, ws))
        for w in ws:
            d = dur * len(w) / total
            words.append({"w": w + " ", "s": offset, "e": offset + d})
            offset += d
        audio.append(a)
    if not audio:
        audio = [np.zeros(SR // 4, dtype=np.float32)]
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(audio), SR, format="WAV")
    out.write(json.dumps({"words": words, "audio": base64.b64encode(buf.getvalue()).decode()}) + "\n")
    out.flush()
