from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import time
import base64
import asyncio
import tempfile
from agent_RC2 import run_conversation  # Import the voice assistant function
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# 📌 Upload PDF and Store in AstraDB
def vector_datastore_upload(pdf_path):
    log_dir = "logs_astraDB"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = os.path.join(log_dir, f"astraDB_log_{timestamp}.txt")

    with open(log_filename, 'w') as log_file:
        embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        vstore_pdf = AstraDBVectorStore(
            collection_name="uploaded",
            embedding=embedding,
            token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        )
        vstore_pdf.clear()
        loader_pdf = PyPDFLoader(pdf_path)
        docs_pdf = loader_pdf.load()
        documents_pdf = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20).split_documents(docs_pdf)
        inserted_ids_pdf = vstore_pdf.add_documents(documents_pdf)
        log_file.write(f"\nInserted {len(inserted_ids_pdf)} documents.")
    return {"message": "Document uploaded and processed!"}

@app.route("/upload-pdf/", methods=["POST"])
def upload_pdf():
    try:
        file = request.files['file']
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        result = vector_datastore_upload(temp_file_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error uploading file: {str(e)}"}), 500

@app.route("/display-pdf/<filename>", methods=["GET"])
def display_pdf(filename):
    pdf_path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(pdf_path):
        return jsonify({"error": "File not found"}), 404
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    return jsonify({"iframe": f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>'})

@app.route("/activate-voice-assistant/", methods=["POST"])
def activate_voice_assistant():
    try:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_filename = os.path.join(log_dir, f"chatbot_log_{timestamp}.txt")
        asyncio.run(run_conversation())
        return jsonify({"message": "Voice Assistant Activated", "log_file": log_filename})
    except Exception as e:
        return jsonify({"error": f"Error activating voice assistant: {str(e)}"}), 500

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
