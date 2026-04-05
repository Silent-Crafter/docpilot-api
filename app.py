import random
import time
import json
import time
import uuid
import base64
from flask import Flask, jsonify, request, stream_with_context, Response
from flask_cors import CORS

from dummyrag import dummyrag

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

chatbot = {}

@app.route('/')
def root():
    return "Api is working"

@app.route('/chat/new', methods=['GET'])
def new_chat():
    uid = str(uuid.uuid4())
    chatbot[uid] = dummyrag()
    timestamp = int(time.time())
    print(f"New chat session created with uuid: {uid}")
    print(f"Current active sessions: {list(chatbot.keys())}")
    return {"chatid": uid, "timestamp": timestamp}

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"type": "error", "content": "Unsupported Media Type"}), 415

    if not data.get('prompt', None):
        return jsonify({"type": "error", "content": "invalid request body. prompt not found"})

    if not data.get('chatid') or data['chatid'] not in chatbot:
        return jsonify({"type": "error", "content": "invalid or missing chatid. call /new_chat first"})

    time.sleep(random.randint(2, 5))

    return jsonify(chatbot[data['chatid']].generate(data['prompt'], stream=False))


@app.route('/stream', methods=['GET'])
def stream():
    prompt = request.args.get("prompt", None)
    session_uuid = request.args.get("chatid", None)

    def sse_error(message):
        err = json.dumps({"type": "error", "content": message})
        def resp(): 
            yield f"data: {err}\n\n"

        return Response(
            stream_with_context(resp()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    if not prompt:
        return sse_error("Please specify a prompt")

    if not session_uuid or session_uuid not in chatbot:
        return sse_error("invalid or missing uuid. call /new_chat first")

    def generator(prompt):
        chat = chatbot.get(session_uuid, None)
        if not chat:
            return sse_error("invalid or missing uuid. call /new_chat first")
        for chunk in chat.generate(prompt, stream=True):
            yield f"data: {json.dumps(chunk)}\n\n"

    return Response(
        stream_with_context(generator(prompt)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
         
@app.route('/stream_p', methods=['POST'])
def stream_post():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"type": "error", "content": "Unsupported Media Type"}), 415

    prompt = data.get('prompt')
    session_uuid = data.get('chatid', None)
    file = data.get('file') # seems unused

    if not prompt:
        return jsonify({"type": "error", "content": "No prompt found"})

    if not session_uuid or session_uuid not in chatbot:
        return jsonify({"type": "error", "content": "invalid or missing uuid. call /new_chat first"})

    def generator(prompt):
        for chunk in chatbot[session_uuid].generate(prompt, stream=True):
            yield f"data: {json.dumps(chunk)}\n\n"

    return Response(
        stream_with_context(generator(prompt)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.route('/history', methods=['POST'])
def history():
    data = request.json
    session_uuid = data.get('chatid', None)

    if not session_uuid or session_uuid not in chatbot:
        return jsonify({"type": "error", "content": "invalid or missing uuid. call /new_chat first"})
    
    return jsonify(chatbot[session_uuid].history)

# Disable Caching
@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == "__main__":
    app.run(host="::", port=31337, debug=False)
