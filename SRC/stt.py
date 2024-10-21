import torch
import torchaudio
import numpy as np
import os
import wave
import pyaudio
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Load Silero VAD model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

def record_audio_with_vad(filename, sample_rate=16000, max_duration=30):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    CHUNK = 512

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=sample_rate, input=True, frames_per_buffer=CHUNK)

    print("Listening... Speak now.")

    frames = []
    vad_iterator = VADIterator(model)
    recording_duration = 0
    is_speech = False

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        audio_chunk = np.frombuffer(data, dtype=np.int16).flatten().astype(np.float32) / 32768.0
        
        speech_dict = vad_iterator(audio_chunk, return_seconds=True)
        
        if speech_dict:
            is_speech = True
            print("Speech detected")
        
        recording_duration += CHUNK / sample_rate

        if recording_duration > max_duration or (is_speech and not speech_dict and recording_duration > 3):
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
if __name__ == "__main__":
    filename = "recorded_audio.wav"
    if record_audio_with_vad(filename):
        transcription_text = transcribe_audio(filename)
        if transcription_text:
            print("Transcription:", transcription_text)
        else:
            print("Transcription failed.")
    else:
        print("No audio was recorded.")