from flask import Flask, render_template, request, jsonify
from src.query_chromadb import process_response_for_api

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question')
    response = process_response_for_api(question)
   
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)