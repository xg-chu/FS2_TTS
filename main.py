import os
import time
import random
import shutil
from synthesizer import Synthesizer
from flask import Flask, request, render_template, jsonify, url_for
app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tts_trans')
def index_tts_trans():
    return render_template('index.html')

model = Synthesizer()
model.load_model()
@app.route('/tts_text', methods=['POST', 'GET'])
def get_tts_text():
    if request.method == 'POST':
        tts_text = request.form.get('tts_text')
        # file_name
        if len(os.listdir('static/tf_outputs')) >= 10:
            shutil.rmtree('static/tf_outputs')
            os.mkdir('static/tf_outputs')
        time_string = time.strftime("%b%d_%H%M_", time.localtime()) + "".join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
        file_path = 'tf_outputs/{}.wav'.format(time_string)
        net_file_path = url_for('static', filename=file_path)
        local_file_path = 'static/' + file_path
        model.synthesis(tts_text, local_file_path)
        #shutil.copyfile('static/audio_before.wav', local_file_path)
        return jsonify({'tts_text': tts_text, 'file_path':net_file_path})

@app.route('/hello')
def hello():
    return 'Hello, World'

if __name__ == '__main__':
    app.run('127.0.0.1', port=5000, debug=True)