from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.playback import play
import sounddevice as sd
from scipy.io.wavfile import write
import os

app = Flask(__name__)

model_size = "distil-large-v3"
fs = 44100  # Sample rate
seconds = 30  # Duration of recording

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # Get the audio file from the request
    audio_file = request.files['audio']
    audio_file.save('temp.wav')
    # Transcribe the audio
    output = transcribe_audio()

    # Remove the temporary audio file
    os.remove('temp.wav')
    return jsonify({'transcription': output})

def transcribe_audio():
    output = ""
    # Record audio
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Audio recording complete , Play Audio")

    # Save as WAV file
    write('temp.wav', fs, myrecording) 

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe("temp.wav", beam_size=5, language="en", condition_on_previous_text=False)
    for segment in segments:
        output += segment.text

    return output

if __name__ == '__main__':
    app.run(debug=True)