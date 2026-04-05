import os
import json
import logging
import uuid
import time

from flask import Flask, request, jsonify, Response, stream_with_context
from werkzeug.datastructures.file_storage import FileStorage
from werkzeug.utils import secure_filename
from rag import ChatBot
from flask_cors import CORS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './data/'

CORS(app, resources={r"/*": {"origins": "*"}})

logger = logging.getLogger(__name__)
chats: dict[str, ChatBot] = {}

@app.route('/')
def root():
    return "Api is running"

@app.route('/chat/new', methods=['POST'])
def new_chat():
    chatid = str(uuid.uuid4())
    chats.update({chatid: ChatBot()})
    timestamp = time.time()

    return jsonify({"chatid": chatid, "timestamp": timestamp})


@app.route('/stream', methods=['GET', 'POST'])
def stream():
    def event_stream(prompt, chatid, timestamp, file_path: str | None = None):
        yield {"type": "START", "status": "", "content": "", "chatid": chatid, "timestamp": timestamp}
        if file_path is not None:
            data = {"type": "file_upload", "status": "Uploading file", "content": file_path}
            yield f"data: {json.dumps(data)}\n\n"
            chats[chatid].add_file(file_path)

        if not prompt.strip():
            yield f"data: {json.dumps({'type': 'answer_with_images', 'content': 'INVALID PROMPT', 'status': 'done'})}\n\n"

        else:
            output = chats['chatid'].generate(prompt)
            for chunk in output:
                yield f"data: {json.dumps(chunk)}\n\n"

    if request.method == 'GET':
        prompt = request.args.get("prompt", "")
        chatid = ''
        timestamp = time.time()
        if request.args.get('chatid') is None:
            chatid = str(uuid.uuid4())
            chats.update({chatid: ChatBot()})

        if chatid not in chats:
            return jsonify({"type": "error", "content": "invalid chatid"})

        return Response( 
            stream_with_context(event_stream(prompt, chatid, timestamp)),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    elif request.method == 'POST':
        data = request.get_json(silent=True)
        file: FileStorage | None = None
        file_path: str | None = None
        if 'file' in request.files:
            file = request.files['file']

        if isinstance(file, FileStorage) and file.filename.strip() != '':
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

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

        return Response( 
            stream_with_context(event_stream(prompt, chatid, file_path, timestamp)),
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

# Disable Caching
@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == '__main__':
    app.run(host='localhost', port=31337, debug=False)
