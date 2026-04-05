import os
import json
import logging
import uuid
import time
import hashlib
from datetime import datetime, timezone

from flask import Flask, request, jsonify, Response, stream_with_context
from werkzeug.datastructures.file_storage import FileStorage
from werkzeug.utils import secure_filename
from rag import ChatBot
from flask_cors import CORS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './data/'

CORS(app, resources={r"/*": {"origins": "*"}})

logger = logging.getLogger(__name__)
chats: dict[str, ChatBot] = {'root': ChatBot()}

def event_stream(prompt, chatid, timestamp, file_path: str | None = None):
    _start = {"type": "START", "status": "", "content": "", "chatid": chatid, "timestamp": timestamp}
    yield f"data: {json.dumps(_start)}\n\n"

    if file_path is not None:
        data = {"type": "file_upload", "status": "Uploading file", "content": file_path}
        yield f"data: {json.dumps(data)}\n\n"
        chats[chatid].add_file(file_path)

    if not prompt.strip():
        yield f"data: {json.dumps({'type': 'answer_with_images', 'content': 'INVALID PROMPT', 'status': 'done'})}\n\n"

    else:
        output = chats[chatid].generate(prompt)
        for chunk in output:
            yield f"data: {json.dumps(chunk)}\n\n"


def sse_error(message):
    err = json.dumps({"type": "error", "content": message})
    def resp(err): 
        yield f"data: {err}\n\n"

    return Response(
        stream_with_context(resp(err)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

# ── Helpers for KnowledgeHub document management ──────────────────────

def _doc_id(filename: str) -> str:
    """Deterministic, URL-safe ID derived from the filename."""
    return hashlib.sha256(filename.encode()).hexdigest()[:16]


def _doc_to_dict(filepath: str) -> dict:
    """Build the document metadata dict the frontend expects."""
    stat = os.stat(filepath)
    filename = os.path.basename(filepath)
    return {
        "id": _doc_id(filename),
        "filename": filename,
        "size": stat.st_size,
        "uploadedAt": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
    }


def _find_doc_by_id(doc_id: str) -> str | None:
    """Resolve hash-based ID back to a filepath, or None."""
    data_dir = app.config['UPLOAD_FOLDER']
    if not os.path.isdir(data_dir):
        return None
    for name in os.listdir(data_dir):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path) and _doc_id(name) == doc_id:
            return path
    return None


# ── Routes ────────────────────────────────────────────────────────────

@app.route('/')
def root():
    return "Api is running"


# ── KnowledgeHub: Document CRUD ──────────────────────────────────────

@app.route('/documents', methods=['GET'])
def list_documents():
    """Return metadata for every file in the data/ folder."""
    data_dir = app.config['UPLOAD_FOLDER']
    os.makedirs(data_dir, exist_ok=True)

    docs = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            docs.append(_doc_to_dict(path))

    return jsonify(docs)


@app.route('/documents', methods=['POST'])
def upload_document():
    """Accept a multipart file upload and save it to data/."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file: FileStorage = request.files['file']
    if not file.filename or file.filename.strip() == '':
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(file.filename)
    data_dir = app.config['UPLOAD_FOLDER']
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)

    file.save(filepath)
    logger.info("KnowledgeHub: uploaded %s", filename)

    # Index the document into the shared vector store
    try:
        chats['root'].add_file(filepath)
        logger.info("KnowledgeHub: indexed %s", filename)
        indexed = True
    except Exception as e:
        logger.error("KnowledgeHub: indexing failed for %s: %s", filename, e)
        indexed = False

    doc = _doc_to_dict(filepath)
    doc["indexed"] = indexed
    return jsonify(doc), 201


@app.route('/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id: str):
    """Delete a document from data/ by its hash-based ID."""
    filepath = _find_doc_by_id(doc_id)
    if filepath is None:
        return jsonify({"error": "Document not found."}), 404

    filename = os.path.basename(filepath)
    os.remove(filepath)
    logger.info("KnowledgeHub: deleted %s", filename)

    return jsonify({"success": True, "id": doc_id})

@app.route('/chat/new')
def new_chat():
    chatid = str(uuid.uuid4())
    chats.update({chatid: ChatBot()})
    timestamp = time.time()

    return jsonify({"chatid": chatid, "timestamp": timestamp})


@app.route('/stream', methods=['GET'])
def stream():
    prompt = request.args.get("prompt", "")
    chatid = request.args.get("chatid", "")
    timestamp = time.time()
    if request.args.get('chatid') is None:
        chatid = str(uuid.uuid4())
        chats.update({chatid: ChatBot()})

    if chats.get(chatid, None) is None:
        return sse_error("invalid chatid")

    return Response( 
        stream_with_context(event_stream(prompt, chatid, timestamp)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@app.route('/stream', methods=['POST'])
def stream_with_file():
    """Stream endpoint with file upload support (multipart/form-data)."""
    prompt = request.form.get('prompt', '')
    chatid = request.form.get('chatid', '')
    timestamp = time.time()

    # Handle optional file upload
    file_path: str | None = None
    if 'file' in request.files:
        file: FileStorage = request.files['file']
        if file.filename and file.filename.strip() != '':
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

    if not prompt:
        return sse_error("No prompt provided")

    # Look up existing chat or create one if no chatid given
    if not chatid:
        chatid = str(uuid.uuid4())
        chats[chatid] = ChatBot()

    if chats.get(chatid) is None:
        return sse_error(f"invalid chatid {chatid}")

    return Response(
        stream_with_context(event_stream(prompt, chatid, timestamp, file_path)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.route('/generate', methods=['POST', 'GET'])
def generate():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'error': 'Invalid request.'}), 415

    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'Invalid request.'}), 400

    chatid = ''
    timestamp = time.time()
    if data.get('chatid') is not None:
        chatid = str(uuid.uuid4())
        chats.update({chatid: ChatBot()})

    if chatid not in chats:
        return jsonify({"type": "error", "content": "invalid chatid"})

    resp = chats[chatid].generate(prompt)
    final_resp = None
    for content in resp:
        final_resp = content

    return jsonify({**final_resp, "chatid": chatid, "timestamp": timestamp})

@app.route('/history', methods=['POST'])
def history():
    data = request.json
    chatid = data.get('chatid', None)

    if chatid is None or chats.get(chatid) is None:
        return jsonify({"type": "error", "content": "invalid or missing uuid. call /new_chat first"})
    
    return jsonify(chats[chatid].get_history())

# Disable Caching
@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == '__main__':
    app.run(host='::', port=31337, debug=False)
