import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from database.db import get_history, save_result
from model.gradcam import generate_heatmap
from model.predict import predict_image
from model.preprocess import detect_face

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/analyze")
def analyze():
    return render_template("analyze.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)
    path = path.replace("\\", "/")

    face = detect_face(path)
    result, confidence = predict_image(face)
    heatmap = None

    if face is not None and result in ("Deepfake", "Morphing Attack"):
        heatmap_name = f"heatmap_{filename}"
        heatmap_path = os.path.join(app.config["UPLOAD_FOLDER"], heatmap_name)
        heatmap = generate_heatmap(path, heatmap_path)

    try:
        save_result(path, result, confidence)
    except Exception as exc:
        return f"Database error: {exc}"

    return render_template(
        "result.html",
        image=path,
        result=result,
        confidence=round(confidence, 2),
        heatmap=heatmap,
    )


@app.route("/history")
def history():
    data = get_history()
    return render_template("history.html", data=data)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
