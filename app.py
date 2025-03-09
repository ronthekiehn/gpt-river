from flask import Flask, render_template, jsonify, request, abort
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import threading
import time
import os
import json
import random
import re

app = Flask(__name__, static_folder='static')

# Initialize GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Simple data storage - just a sequence and text
DATA_FILE = 'river_data.json'
MAX_LENGTH = 3500  # Max chars to keep
current_sequence = 0
user_text = []
file_lock = threading.Lock()  # Simple lock for file writes only
generation_lock = threading.Lock()  # Lock to prevent overlapping generations
is_generating = False  # Flag to track active generation
fallback_text = "Once upon a time..."

# Rate limiting and validation
RATE_LIMIT_SECONDS = 3  # Time between allowed contributions
MAX_WORD_LENGTH = 15
contribution_timestamps = {}  # IP -> last contribution time

# Initialize data file
def init_data_file():
    global current_sequence
    if not os.path.exists(DATA_FILE):
        with file_lock:  # Only lock for writing
            with open(DATA_FILE, 'w') as f:
                json.dump({
                    'text': fallback_text,
                    'sequence': 0,
                    'new_text': fallback_text
                }, f)
        current_sequence = 0
    else:
        # Load existing sequence if file exists
        try:
            with open(DATA_FILE, 'r') as f:
                data = json.load(f)
                current_sequence = data.get('sequence', 0)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading data file: {e}")
            # Create a new file if there's a problem
            with file_lock:
                with open(DATA_FILE, 'w') as f:
                    json.dump({
                        'text': fallback_text,
                        'sequence': 0,
                        'new_text': fallback_text
                    }, f)
            current_sequence = 0

init_data_file()

# Read the current data from the file - no locking for reads
def read_data_file():
    try:
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading data file: {e}")
        return {'text': "", 'sequence': 0, 'new_text': ""}

# Update the river text with proper sequence management
def update_river_text(full_text, new_text):
    global current_sequence
    
    # Lock only for writing
    with file_lock:
        # Read current data first to ensure we have latest sequence
        try:
            with open(DATA_FILE, 'r') as f:
                current_data = json.load(f)
                # Use the latest sequence from the file
                file_sequence = current_data.get('sequence', 0)
        except Exception:
            file_sequence = current_sequence
        
        # Always increment from the highest known sequence
        next_sequence = max(current_sequence, file_sequence) + 1
        current_sequence = next_sequence
        
        # Trim to max length
        if len(full_text) > MAX_LENGTH:
            full_text = full_text[-MAX_LENGTH:]
        
        # Save to file
        data = {
            'text': full_text,
            'sequence': next_sequence,
            'new_text': new_text
        }
        
        with open(DATA_FILE, 'w') as f:
            json.dump(data, f)
        
        return data

def generate_text():
    """Background thread that generates text periodically"""
    global is_generating
    
    def sanitize_text(text):
        # Normalize text for smoother display
        text = ''.join(char for char in text if char.isprintable() and (char.isalnum() or char in ' .,!?-'))
        return text.strip()
    
    counter = 0
    
    while True:
        start_time = time.time()
        # Only start a new generation if we're not already generating
        if generation_lock.acquire(blocking=False):  # Non-blocking attempt to acquire lock
            try:
                is_generating = True
                print(f"Generating text..., loop {counter}")
                counter += 1
                
                # Get current text
                data = read_data_file()
                current_text = data['text']
                # Get last 1000 characters as context
                
                context = current_text[-1000:].replace('[[', '').replace(']]', '')
                # Generate new text                
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
                except Exception as e: # Try again with fallback text
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
                
                #print(f"Generated text: {new_text}")
                # Only keep the first 80 characters
                new_text = new_text[:78]
                #print(f"Generated text: {new_text}")
                # Add user contributions if available
                user_contributions = []
                with threading.Lock():
                    if user_text:
                        user_contributions = user_text.copy()
                        user_text.clear()
                
                if user_contributions:
                    words = new_text.split()
                    insert_positions = sorted(random.sample(range(len(words)), len(user_contributions)))
                    for pos, text in zip(insert_positions, user_contributions):
                        words.insert(pos, text)
                    
                    new_text = " ".join(words)
                
                # Add to current text and update
                result = update_river_text(current_text + ' ' + new_text, new_text)
                print(f"Text generated successfully - Sequence: {result['sequence']}")
                #print(f"Time taken: {time.time() - start_time:.2f}s")
                
            except Exception as e:
                print(f"Error generating text: {e}")
            finally:
                # Always mark as not generating and release the lock when done
                is_generating = False
                generation_lock.release()
        
        # Sleep between generation attempts - outside the lock
        time.sleep(3)  # sleep + 1s for generation time

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text')
def get_text():
    try:
        data = read_data_file()
        return jsonify(data)
    except Exception as e:
        return jsonify({'text': f"Error: {e}", 'sequence': -1, 'new_text': ""})

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
    global contribution_timestamps
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
                if now - timestamp < 3600  # Keep last hour
            }
            
        with threading.Lock():
            user_text.append(f'[[{word}]]')
        
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

# Fix for Flask debug mode which runs two instances
# Only start the thread in the main process, not in Flask's reloader process
# Also works for regular production mode
import os
if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not os.environ.get('FLASK_DEBUG'):
    print("Main process detected, starting background thread")
    start_background_thread()
else:
    print("Reloader or initialization process detected, not starting background thread")

# Remove the app.app_context() call which doesn't fix the issue
# Keep this for direct execution only if needed
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
