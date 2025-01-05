from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
import time
from src.query_chromadb import process_query, get_random_document_chunks
from src.create_knowledge_bank import store_file_in_chromadb_txt_file

# Configurations
class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'default_secret_key')  # Replace with a secure key in production
    SQLALCHEMY_DATABASE_URI = 'sqlite:///chat_history.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DEBUG = True

app = Flask(__name__, template_folder=os.path.abspath('src/templates'), static_folder=os.path.abspath('src/static'))
app.config.from_object(Config)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Models
class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    designation = db.Column(db.String(100), nullable=False)

class ChatHistory(db.Model):
    __tablename__ = 'chat_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    follow_ups = db.Column(db.Text)
    processing_time = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

@login_manager.user_loader
def load_user(user_id):
    return User.query.filter_by(id=user_id).first()

# Helper functions
def save_chat_history(question, response, processing_time, user_id):
    try:
        chat = ChatHistory(
            user_id=user_id,
            question=question,
            answer=response['answer'],
            follow_ups=json.dumps(response.get('follow_ups', [])),
            processing_time=processing_time
        )
        db.session.add(chat)
        db.session.commit()
    except Exception as e:
        app.logger.error(f"Error saving chat history: {e}")

# Routes
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            password = generate_password_hash(request.form['password'])
            designation = request.form['designation']
            
            if User.query.filter_by(email=email).first():
                flash('Email already exists. Please use a different email.', 'danger')
                return redirect(url_for('signup'))
            
            user = User(name=name, email=email, password=password, designation=designation)
            db.session.add(user)
            db.session.commit()
            flash('Signup successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            app.logger.error(f"Error during signup: {e}")
            flash('An error occurred. Please try again.', 'danger')
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    return render_template('index.html')

@app.route('/random_questions')
@login_required
def random_questions():
    try:
        questions = get_random_document_chunks()
        return jsonify(questions)
    except Exception as e:
        app.logger.error(f"Error fetching random questions: {e}")
        return jsonify({'error': 'Unable to fetch random questions.'}), 500

@app.route('/chat')
@login_required
def chat():
    return render_template('chat.html', name=current_user.name)

@app.route('/chat_history', methods=['GET'])
@login_required
def chat_history():
    try:
        history = ChatHistory.query.filter_by(user_id=current_user.id).order_by(ChatHistory.timestamp.desc()).all()
        chat_data = [{
            'question': chat.question,
            'answer': chat.answer,
            'follow_ups': json.loads(chat.follow_ups),
            'processing_time': chat.processing_time,
            'timestamp': chat.timestamp
        } for chat in history]
        return jsonify(chat_data)
    except Exception as e:
        app.logger.error(f"Error fetching chat history: {e}")
        return jsonify({'error': 'Unable to fetch chat history.'}), 500

@app.route('/ask', methods=['POST'])
@login_required
def ask():
    try:
        start_time = time.time()
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Question cannot be empty or null.'}), 400
        
        question = data['question'].strip().lower()
        number_of_results = data.get('number_of_results', 5)
        is_rephrased = data.get('is_rephrased', True)

        if not question:
            return jsonify({'error': 'Question cannot be empty or null.'}), 400
        
        chat = ChatHistory.query.filter_by(user_id=current_user.id, question=question).first()
        if chat:
            response = {
                'answer': chat.answer,
                'follow_ups': json.loads(chat.follow_ups),
                'processing_time': time.time() - start_time,
                'source': 'history'
            }
            return jsonify(response)
        
        response = process_query(question, number_of_results=number_of_results,
                                 is_rephrased=is_rephrased)
        processing_time = time.time() - start_time
        save_chat_history(question, response, processing_time, current_user.id)
        return jsonify({**response, 'processing_time': processing_time, 'source': 'generated'})
    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify({'error': 'An error occurred while processing your question.'}), 500

@app.route('/store_data', methods=['GET'])
@login_required
def store_data():
    try:
        data_dir = "src/input_data"
        filenames = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        store_file_in_chromadb_txt_file(data_dir, filenames)
        return jsonify({'status': 'success'})
    except Exception as e:
        app.logger.error(f"Error storing data: {e}")
        return jsonify({'error': 'Unable to store data.'}), 500

@app.route('/clear_chat_history', methods=['GET'])
@login_required
def clear_chat_history():
    try:
        ChatHistory.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        return jsonify({'status': 'success'})
    except Exception as e:
        app.logger.error(f"Error clearing chat history: {e}")
        return jsonify({'error': 'Unable to clear chat history.'}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Initialize database tables
    app.run(host='0.0.0.0', port=5000)
