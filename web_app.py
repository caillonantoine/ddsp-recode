from flask import Flask, request, url_for, redirect, send_from_directory
import os
import torch
from ddsp.core import extract_loudness, extract_pitch
import librosa as li
import numpy as np
import soundfile as sf

os.makedirs("tmp", exist_ok=True)
UPLOAD_FOLDER = "./tmp/"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = torch.jit.load("export/ddsp_saxoo_pretrained.ts")
sampling_rate = model.ddsp.sampling_rate.item()
block_size = model.ddsp.block_size.item()


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        file.save(os.path.join(UPLOAD_FOLDER, "audio.wav"))

        x, sr = li.load(
            os.path.join(UPLOAD_FOLDER, "audio.wav"),
            sampling_rate,
        )

        N = (sampling_rate - (len(x) % sampling_rate)) % sampling_rate
        x = np.pad(x, (0, N))

        f0 = extract_pitch(x, 16000, 160)
        lo = extract_loudness(x, 16000, 160)
        with torch.no_grad():
            f0 = torch.from_numpy(f0).reshape(1, -1, 1).float()
            lo = torch.from_numpy(lo).reshape(1, -1, 1).float()

            y = model(f0, lo).reshape(-1).numpy()

        sf.write("./tmp/audio.wav", y, sampling_rate)

        return send_from_directory("tmp", "audio.wav", as_attachment=True)

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''