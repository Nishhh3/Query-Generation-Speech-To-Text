import os, uuid, ffmpeg
import pandas as pd
import numpy as np
import mysql.connector
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import random

# Load env
load_dotenv()
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ASR_MODEL = os.getenv("ASR_MODEL", "openai/whisper-small")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
USE_GPU = int(os.getenv("USE_GPU", "0"))
SIM_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))

# MySQL connection
db = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
    database=os.getenv("DB_NAME")
)

# Load dataset
df = pd.read_csv("banking_dataset.csv")
keyword_texts = df["keywords"].astype(str).tolist()

# Load models
device = 0 if USE_GPU else -1
asr = pipeline("automatic-speech-recognition", model=ASR_MODEL, device=device)
embedder = SentenceTransformer(EMBED_MODEL)

# Precompute embeddings
EMBED_CACHE = "keyword_embeddings.npy"
if os.path.exists(EMBED_CACHE):
    keyword_embeddings = torch.tensor(np.load(EMBED_CACHE))
else:
    keyword_embeddings = embedder.encode(keyword_texts, convert_to_tensor=True)
    np.save(EMBED_CACHE, keyword_embeddings.cpu().numpy())

# Helpers
def convert_to_wav(input_path, out_path):
    ffmpeg.input(input_path).output(out_path, ac=1, ar="16000").run(overwrite_output=True)

def asr_from_file(path):
    wav_path = path
    if not path.endswith(".wav"):
        wav_path = os.path.splitext(path)[0] + ".wav"
        convert_to_wav(path, wav_path)
    res = asr(wav_path)
    return res.get("text", "").strip()

def categorize_complaint(text):
    emb = embedder.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(emb, keyword_embeddings)[0]
    best_idx = int(torch.argmax(scores).item())
    best_score = float(scores[best_idx].item())
    if best_score < SIM_THRESHOLD:
        return "Unknown", "Uncategorized"
    row = df.iloc[best_idx]
    return row.get("department", "Unknown"), row.get("category", "Uncategorized")

def generate_ticket_code():
    """Generate a random unique 5-digit ticket code."""
    while True:
        code = f"{random.randint(10000, 99999)}"  # 5-digit random number
        cursor = db.cursor()
        cursor.execute("SELECT 1 FROM tickets WHERE ticket_code=%s", (code,))
        exists = cursor.fetchone()
        if not exists:
            return code



# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/tickets", methods=["GET"])
def get_tickets():
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT ticket_code, input_type, transcript, department, category, status FROM tickets")
    rows = cursor.fetchall()
    return jsonify(rows)


@app.route("/submit", methods=["POST"])
def submit():
    input_type = request.form.get("type")
    transcript, saved_file_path = "", None

    if input_type in ["video", "audio"]:
        f = request.files.get("file")
        if not f: 
            return jsonify({"error": "No file uploaded"}), 400
        filename = secure_filename(f.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        save_path = os.path.join(UPLOAD_FOLDER, unique_name)
        f.save(save_path)
        audio_path = os.path.splitext(save_path)[0] + ".wav"
        convert_to_wav(save_path, audio_path)
        transcript = asr_from_file(audio_path)
        saved_file_path = save_path

    elif input_type == "text":
        transcript = request.form.get("text", "").strip()
        if not transcript: 
            return jsonify({"error": "No text provided"}), 400
        saved_file_path = None
    else:
        return jsonify({"error": "Invalid input type"}), 400

    # Categorize complaint
    dept, cat = categorize_complaint(transcript)

    # Generate unique 5-digit ticket code
    ticket_code = generate_ticket_code()

    # Insert ticket into DB
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO tickets (ticket_code, input_type, file_path, transcript, department, category) VALUES (%s, %s, %s, %s, %s, %s)",
        (ticket_code, input_type, saved_file_path, transcript, dept, cat)
    )
    db.commit()

    return jsonify({
        "message": "Ticket submitted successfully",
        "ticket_id": ticket_code,
        "transcript": transcript,
        "department": dept,
        "category": cat
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
