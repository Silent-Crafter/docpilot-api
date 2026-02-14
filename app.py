import json
from flask import Flask, request, jsonify, make_response, Response, stream_with_context
from rag import ChatBot
from flask_cors import CORS

app = Flask(__name__)
chatbot = ChatBot()

CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def root():
    return "Api is running"


@app.route('/stream')
def stream():
    prompt = request.args.get("prompt", "")

    def event_stream():
        nonlocal prompt
        if not prompt.strip():
            yield f"data: {json.dumps({'type': 'answer_with_images', 'content': 'try again', 'status': 'done'})}\n\n"

        else:
            output = chatbot.generate(prompt)
            while True:
                try:
                    chunk = next(output)
                    if chunk.get('type') == 'answer_with_images':
                        __import__('time').sleep(1.5)
                    yield f"data: {json.dumps(chunk)}\n\n"
                except StopIteration:
                    break

    return Response( 
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.route('/generate', methods=['POST', 'GET'])
def generate():
    data: dict = request.get_json(silent=True)
    print(data)
    if not data:
        return jsonify({'error': 'Invalid request.'}), 415

    prompt = data.get('prompt')
    print(prompt)
    if not prompt:
        return jsonify({'error': 'Invalid request.'}), 400

    resp = chatbot.generate(prompt)
    final_resp = None
    while True:
        try:
            content = next(resp)
            final_resp = content
        except StopIteration:
            break

    return jsonify(final_resp)

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
