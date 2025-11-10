import os, io, base64, wave, struct, math

def tts_to_base64(text: str):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        for i in range(16000):
            val = int(32767.0*0.3*math.sin(2.0*math.pi*440.0*(i/16000.0)))
            wf.writeframesraw(struct.pack("<h", val))
    return {"ok": True, "mime":"audio/wav", "audio_b64": base64.b64encode(buf.getvalue()).decode()}

def stt_from_wav_bytes(wav_bytes: bytes, model: str = "base"):
    return {"ok": False, "err": "Whisper not installed in this bundle. Install openai-whisper for STT."}
