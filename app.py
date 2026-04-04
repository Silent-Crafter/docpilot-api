import random
import time
import json
import base64

from flask import Flask, jsonify, request, stream_with_context, Response
from flask_cors import CORS

from dummyrag import dummyrag

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

chatbot = dummyrag()

@app.route('/')
def root():
    return "Api is working"


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"type": "error", "content": "Unsupported Media Type"}), 415

    if not data.get('prompt', None):
        return jsonify({"type": "error", "content": "invalid request body. prompt not found"})

    time.sleep(random.randint(2, 5))

    return jsonify(chatbot.generate(data['prompt'], stream=False))


@app.route('/stream', methods=['GET'])
def stream():
    prompt = request.args.get("prompt", None)
    if not prompt:
        err = json.dumps({"type": "error", "content": "Please specify a prompt"})
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

    def generator(prompt):
        for chunk in chatbot.generate(prompt, stream=True):
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
    file = data.get('file')

    if not prompt:
        return jsonify({"type": "error", "content": "No prompt found"})

    def generator(prompt):
        for chunk in chatbot.generate(prompt, stream=True):
            yield f"data: {json.dumps(chunk)}"

    return Response(
        stream_with_context(generator(prompt)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

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
