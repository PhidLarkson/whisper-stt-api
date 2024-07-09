import sounddevice as sd
from scipy.io.wavfile import write
import requests

# Set the API endpoint URL
api_url = "https://whisper-stt-api.onrender.com/transcribe"

# Set the recording parameters
sample_rate = 16000
duration = 10  # seconds

def record_audio():
    """Record audio for the specified duration"""
    print("Recording audio...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    return audio

def send_audio_to_api(audio):
    """Send the recorded audio to the Whisper STT API"""
    # Convert the audio to a WAV file
    write('recording.wav', sample_rate, audio)
    with open('recording.wav', 'rb') as f:
        files = {'audio': f}
        response = requests.post(api_url, files=files)
        if response.status_code == 200:
            return response.json()['transcription']
        else:
            return "Error: Failed to transcribe audio"

def main():
    audio = record_audio()
    transcription = send_audio_to_api(audio)
    print(f"Transcription: {transcription}")

if __name__ == "__main__":
    main()