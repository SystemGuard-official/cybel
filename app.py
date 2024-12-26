from flask import Flask, render_template, request, jsonify
import sqlite3
from functools import lru_cache
import time
import json
import os
from src.query_chromadb import process_response_for_api
from src.create_knowledge_bank import store_file_in_chromadb_txt_file

app = Flask(__name__, 
    template_folder=os.path.abspath('src/templates'),
    static_folder=os.path.abspath('src/static'))

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  question TEXT NOT NULL,
                  answer TEXT NOT NULL,
                  follow_ups TEXT,
                  processing_time FLOAT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# Database helper functions
def get_db_connection():
    return sqlite3.connect('chat_history.db')

def save_to_db(question, response, processing_time):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''INSERT INTO chat_history (question, answer, follow_ups, processing_time)
                     VALUES (?, ?, ?, ?)''',
                  (question, 
                   response['answer'],
                   json.dumps(response.get('follow_ups', [])),
                   processing_time))
        conn.commit()
    finally:
        conn.close()

def get_from_db(question):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''SELECT answer, follow_ups, processing_time, timestamp 
                     FROM chat_history 
                     WHERE question = ? 
                     ORDER BY timestamp DESC 
                     LIMIT 1''', (question,))
        result = c.fetchone()
        return result
    finally:
        conn.close()

def get_all_chats():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''SELECT question, timestamp FROM chat_history ORDER BY timestamp DESC''')
        result = c.fetchall()
        return result
    finally:
        conn.close()
    
# LRU Cache for storing recent results
@lru_cache(maxsize=2)
def get_cached_response(question):
    return process_response_for_api(question)

@app.route('/chat_history')
def chat_history():
    chat_history = get_all_chats()
    return jsonify(chat_history)

@app.route('/clear_chat_history')
def clear_chat_history():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''DELETE FROM chat_history''')
        conn.commit()
    finally:
        conn.close()
    return jsonify({'status': 'success'})

@app.route('/')
def home():
    chat_history = get_all_chats()
    print(chat_history)
    return render_template('claude.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question')
    start_time = time.time()

    # Check database
    db_result = get_from_db(question)
    if db_result:
        processing_time = time.time() - start_time
        return jsonify({
            'answer': db_result[0],
            'follow_ups': json.loads(db_result[1]),
            'processing_time': processing_time,
            'timestamp': db_result[3],
            'source': 'database'
        })

    # Process new response
    response = process_response_for_api(question)
    processing_time = time.time() - start_time

    # Save to database
    save_to_db(question, response, processing_time)

    return jsonify({**response, 'processing_time': processing_time, 'source': 'new'})

@app.route('/store_data', methods=['GET'])
def store_data():
    data_dir = "src/input_data"
    filenames = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    store_file_in_chromadb_txt_file(data_dir, filenames)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
