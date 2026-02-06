# app.py (patched)
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# import the prediction helper from your model.py
from model import predict_video_from_path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Primary predict route
@app.route("/predict", methods=["POST"])
def predict_route():
    if "video" not in request.files:
        return jsonify({"error":"no file part 'video'"}), 400
    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error":"no selected file"}), 400

    filename = secure_filename(f.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(save_path)

    try:
        result = predict_video_from_path(save_path)
    finally:
        try: os.remove(save_path)
        except: pass

    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)

# Compatibility route (some front-ends call /predict_video)
@app.route("/predict_video", methods=["POST"])
def predict_video_compat():
    # reuse same logic as /predict
    if "video" not in request.files:
        return jsonify({"error":"no file part 'video'"}), 400
    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error":"no selected file"}), 400

    filename = secure_filename(f.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(save_path)

    try:
        result = predict_video_from_path(save_path)
    finally:
        try: os.remove(save_path)
        except: pass

    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)

if __name__ == "__main__":
    # debug=True for development only; set to False in production
    app.run(host="0.0.0.0", port=5000, debug=True)
