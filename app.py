from flask import Flask, render_template, jsonify, request, abort
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import threading
import time
import os
import random
import re
import queue
from threading import Lock

app = Flask(__name__, static_folder='static')

# Initialize GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

fallback_text = "Once upon a time..."

# In-memory storage with thread-safe access
class RiverStorage:
    def __init__(self):
        self.lock = Lock()
        self.text = fallback_text
        self.sequence = 0
        self.new_text = fallback_text
        self.MAX_LENGTH = 3500

    def update(self, full_text, new_text):
        with self.lock:
            if len(full_text) > self.MAX_LENGTH:
                full_text = full_text[-self.MAX_LENGTH:]
            self.text = full_text
            self.new_text = new_text
            self.sequence += 1
            return {
                'text': self.text,
                'sequence': self.sequence,
                'new_text': self.new_text
            }

    def get_current(self):
        # No lock needed for reads since they're atomic
        return {
            'text': self.text,
            'sequence': self.sequence,
            'new_text': self.new_text
        }

# Global storage instance
river_storage = RiverStorage()

# Other constants and storage
user_text = queue.Queue()
generation_lock = Lock()
user_lock = Lock()
is_generating = False
RATE_LIMIT_SECONDS = 4
MAX_WORD_LENGTH = 15
contribution_timestamps = {}
interval = 3.5

def generate_text():
    """Background thread that generates text periodically"""
    global is_generating
    
    def sanitize_text(text):
        text = ''.join(char for char in text if char.isprintable() and (char.isalnum() or char in ' .,!?-'))
        return text.strip()
    
    while True:
        if generation_lock.acquire(blocking=False):
            try:
                start_time = time.time()
                is_generating = True
                
                # Get current text from memory
                current_data = river_storage.get_current()
                current_text = current_data['text']

                # Get last 2000 characters as context
                context = current_text[-2000:].replace('[[', '').replace(']]', '')
                
                try:
                    inputs = tokenizer(context, return_tensors='pt')
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=30,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=40,
                        no_repeat_ngram_size=2,
                        pad_token_id=tokenizer.eos_token_id,
                        attention_mask=inputs.attention_mask
                    )
                    new_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                except Exception:
                    context = fallback_text
                    inputs = tokenizer(context, return_tensors='pt')
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=30,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=40,
                        no_repeat_ngram_size=2,
                        pad_token_id=tokenizer.eos_token_id,
                        attention_mask=inputs.attention_mask
                    )
                    new_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Only keep the newly generated part
                if len(context) > 0 and new_text.startswith(context):
                    new_text = new_text[len(context):]

                # Sanitize the generated text
                new_text = sanitize_text(new_text)
                
                new_text = new_text.replace('[[', '').replace(']]', '')
                
                # Fallback on empty text
                if not new_text.strip():
                    print("Empty text generated, falling back...")
                    context = fallback_text
                    inputs = tokenizer(context, return_tensors='pt')
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=30,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=40,
                        no_repeat_ngram_size=2,
                        pad_token_id=tokenizer.eos_token_id,
                        attention_mask=inputs.attention_mask
                    )
                    new_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Only keep the first 80 characters
                new_text = new_text[:78]
                
                # Add user contributions if available
                user_contributions = []
                with user_lock:
                    while not user_text.empty():
                        user_contributions.append(user_text.get())
                
                if user_contributions:
                    words = new_text.split()
                    insert_positions = sorted(random.sample(range(len(words)), len(user_contributions)))
                    for pos, text in zip(insert_positions, user_contributions):
                        words.insert(pos, text)
                    
                    new_text = " ".join(words)
                
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                print(f"Generated text in {elapsed:.2f}s, sleeping for {sleep_time:.2f}s")
                # Update in-memory storage
                river_storage.update(current_text + ' ' + new_text, new_text)
                
            except Exception as e:
                print(f"Error generating text: {e}")
            finally:
                is_generating = False
                generation_lock.release()
        
        time.sleep(sleep_time)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text')
def get_text():
    return jsonify(river_storage.get_current())

# Add input validation function
def validate_word(word):
    # Check length
    if not 1 <= len(word) <= MAX_WORD_LENGTH:
        return False
        
    # Blacklist of dangerous characters/sequences
    dangerous_patterns = [
        '<', '>', 'script', 'onclick', 'onerror', 'onload',  # XSS prevention
        'javascript:', 'data:', 'vbscript:',                  # Protocol handlers
        '\\', '&quot;', '&#', '\\u', '\\x',                  # Escapes and entities
        'eval(', 'setTimeout', 'setInterval',                 # JS functions
        'document.', 'window.'                               # DOM access
    ]
    
    word_lower = word.lower()
    return not any(pattern in word_lower for pattern in dangerous_patterns)

@app.route('/contribute', methods=['POST'])
def contribute():
    global contribution_timestamps, user_text
    try:
        # Get client IP
        client_ip = request.remote_addr
        
        # Check rate limit
        now = time.time()
        if client_ip in contribution_timestamps:
            time_since_last = now - contribution_timestamps[client_ip]
            if time_since_last < RATE_LIMIT_SECONDS:
                return jsonify({
                    'success': False, 
                    'message': f'Please wait {RATE_LIMIT_SECONDS - int(time_since_last)} seconds'
                })

        data = request.json
        word = data.get('word', '').strip()

        # Validate input
        if not validate_word(word):
            return jsonify({
                'success': False,
                'message': 'Invalid word format'
            })
            
        # Update rate limit timestamp
        contribution_timestamps[client_ip] = now

        # Clean up old rate limit entries periodically
        if len(contribution_timestamps) > 10000:  # Prevent memory growth
            now = time.time()
            contribution_timestamps = {
                ip: timestamp 
                for ip, timestamp in contribution_timestamps.items() 
                if now - timestamp < 600  # Keep last 10 minutes
            }
            
        with user_lock:
            user_text.put(f'[[{word}]]')
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# Global flag to track if thread is already started
thread_started = False

def start_background_thread():
    global thread_started
    if thread_started:
        print("Background thread already started, skipping...")
        return
        
    thread = threading.Thread(target=generate_text, daemon=True)
    thread.start()
    thread_started = True
    print(f"Background thread started successfully with ID: {thread.ident}")

if "GUNICORN_CMD_ARGS" in os.environ or not os.environ.get('FLASK_DEBUG'):
    print("Gunicorn master process detected, starting background thread")
    start_background_thread()
else:
    print("Reloader or initialization process detected, not starting background thread")
