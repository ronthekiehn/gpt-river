# gpt-river

A testimate to how far language models have come. GPT River is an interactive art piece that generates a continuous flow of text using GPT-2. Users can contribute individual words that will be incorporated into the generated text stream, as well as GPT-2's context.

**Check it out:** [gptriver.ronkiehn.dev](https://gptriver.ronkiehn.dev)

<img width="1191" alt="image" src="https://github.com/user-attachments/assets/65f4d471-42ea-4f3d-b581-0bf86686ca51" />


## Stack

- Flask + Huggingface Transfomers
- Uses GPT-2 (124M)
- Vanilla HTML CSS JS
- Deployed using Coolify (shoutout!)

## Host it locally!?

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/ronthekiehn/gpt-river.git
   cd gpt-river
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies using requirements.txt:
   ```
   pip install -r requirements.txt
   ```

4. The first time you run the application, it will download the GPT-2 model (~500MB).

### Running the Application

#### Option 1: Using Flask (Development)

```
flask run --host=0.0.0.0 --port=5000
```

#### Option 2: Using Gunicorn (Production)

```
gunicorn --threads 4 -w 1 -b 0.0.0.0:5000 app:app
```

**Note**: The application is designed to work with a single worker (`-w 1`) since it uses in-memory storage. Multiple workers would create separate instances of the text generator, or lose user contributions.

### Accessing the Application

Once running, access the application in your web browser at:

```
http://localhost:5000
```

## Customization

- To change the base text prompt, modify the `fallback_text` variable in `app.py`
- To adjust text generation parameters like temperature, edit the `generate_text` function
- To modify the UI, edit the `templates/index.html` file


## Acknowledgments

- Claude 3.5 and 3.7 are my and helped a lot :) 
- Lo-fi beats provided by [Lofi Girl - lofi hip hop radio 📚 - beats to relax/study to](https://www.youtube.com/watch?v=jfKfPfyJRdk)
