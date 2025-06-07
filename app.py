from flask import Flask, render_template, request, send_file
import whisper
from transformers import pipeline
import os

app = Flask(__name__)
model = whisper.load_model("base")
summarizer = pipeline("summarization", model="t5-base")

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    transcript = summary = ""
    if request.method == "POST":
        file = request.files["audio"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        result = model.transcribe(filepath)
        transcript = result["text"]

        summary_result = summarizer(transcript, max_length=100, min_length=30, do_sample=False)
        summary = summary_result[0]["summary_text"]

        # Save files
        with open("transcript.txt", "w") as f: f.write(transcript)
        with open("summary.txt", "w") as f: f.write(summary)

    return render_template("index.html", transcript=transcript, summary=summary)

@app.route("/download/<filename>")
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
