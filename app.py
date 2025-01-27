import json
from flask import Flask, request, jsonify, make_response, Response
from rag import ChatBot

app = Flask(__name__)
chatbot = ChatBot()

@app.route('/')
def root():
    return "Api is running"


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

    return jsonify(resp)

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
