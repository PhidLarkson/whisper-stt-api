from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import os

app = Flask(__name__)

model_size = "distil-large-v3"

# @app.route('/transcribe', methods=['POST'])
def transcribe():
    # Get the audio data from the request
    audio_data = request.files['audio'].read()

    # Save the audio data to a temporary file
    with open('temp.wav', 'wb') as f:
        f.write(audio_data)

    # Transcribe the audio
    output = transcribe_audio('temp.wav')

    # Remove the temporary audio file
    os.remove('temp.wav')

    # Return the transcription as JSON
    return jsonify({'transcription': output})

def transcribe_audio(audio_file):
    output = ""
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_file, beam_size=5, language="en", condition_on_previous_text=False)
    for segment in segments:
        output += segment.text
    return output

if __name__ == '__main__':
    app.run(debug=True)