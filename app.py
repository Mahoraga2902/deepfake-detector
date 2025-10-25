from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
from model import predict_image, predict_video
import os

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}
UPLOAD_FOLDER = "uploads"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_ext(filename, allowed):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_ext(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({"error": "Unsupported file type. Please upload an image."}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        prediction = predict_image(save_path)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": "Prediction error", "details": str(e)}), 500

@app.route("/predict_video", methods=["POST"])
def predict_video_route():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_ext(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({"error": "Unsupported file type. Please upload a video."}), 400

    sample_every_sec = float(request.form.get("sample_every_sec", 1.0))
    max_frames = int(request.form.get("max_frames", 64))

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        result = predict_video(save_path, sample_every_sec=sample_every_sec, max_frames=max_frames)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": "Prediction error", "details": str(e)}), 500

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
