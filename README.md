# FS2_TTS
TTS system base on FastSpeech2 and MelGAN. 

# Build
1. Requirements:
    * python 3.8.5, flask 1.1.2, tensorflow 2.3.1, react v17.0.1, node v14.15.1, pypinyin 0.40.0, soundfile 0.10.3.post1, scipy 1.5.4
2. Environments:
    * Build virtual environments and install flask.
    ```
    mkdir CN_FS2_TTS
    cd CN_FS2_TTS
    python3 -m venv flask_venv
    . flask_venv/bin/activate
    pip install Flask
    ```
    * Build React environments.
    ```
    npx create-react-app react_front_end
    cd react_front_end/src
    rm -f *
    cd ../
    cp ../react_front_end_src/* src/
    npm install @material-ui/core
    npm install --save react-router-dom
    ```
    and add 
    ```
    "prebuild": "rm ../templates/index.html && rm -rf ../static",
    "postbuild": "mv build/index.html ../templates/ && mv build/static ../static && mkdir ../static/tf_outputs",
    ```
    to "scripts" in 'package.json' in react_front_end.
