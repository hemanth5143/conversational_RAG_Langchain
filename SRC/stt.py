import pyaudio
import wave
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def record_audio(filename, sample_rate=16000, silence_threshold=300, silence_duration=1.0, min_duration=2.0):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    CHUNK = 1024

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=sample_rate, input=True, frames_per_buffer=CHUNK)

    print("Listening... Speak now.")

    frames = []
    silent_chunks = 0
    audio_started = False
    recording_duration = 0

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.int16)
        volume_norm = np.linalg.norm(audio_data) * 10

        if volume_norm > silence_threshold:
            silent_chunks = 0
            audio_started = True
        elif audio_started:
            silent_chunks += 1

        recording_duration += CHUNK / sample_rate

        if audio_started and silent_chunks > int(silence_duration * sample_rate / CHUNK) and recording_duration > min_duration:
            break

        if recording_duration > 30:  # Maximum recording duration of 30 seconds
            break

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    if len(frames) == 0:
        print("No audio detected.")
        return False

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Audio saved to {filename}")
    return True

def transcribe_audio(file_path):
    client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))
    
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
    return transcription.text

# Main execution
filename = "recorded_audio.wav"
if record_audio(filename):
    transcription_text = transcribe_audio(filename)
    if transcription_text:
        print("Transcription:", transcription_text)
    else:
        print("Transcription failed.")
else:
    print("No audio was recorded.")