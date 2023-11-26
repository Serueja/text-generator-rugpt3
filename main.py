from fastapi import FastAPI, Request
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from fastapi.templating import Jinja2Templates

app = FastAPI()

@app.post('/generate_text')
def generate_text(payload: dict[str, str]):
    # Retrieve the input text from the payload
    input_text = payload.get('text')
    
    # Initialize the generator
    generator = pipeline('text-generation', model='ai-forever/rugpt3medium_based_on_gpt2')
    # Generate text using the GPT model
    if input_text.split()[0].lower() == 'аннотация' :
        config_annotation = {
            "max_length": 250,
            "min_length":100,
            "temperature": 1.1,
            "top_p": 2.,
            "num_beams": 10,
            "repetition_penalty": 1.5,
            "num_return_sequences": 4,
            "no_repeat_ngram_size": 2,
            "do_sample": True
        }
        text = generator(input_text, pad_token_id=50256, **config_annotation)
    else:
        config_intro = {
            "max_length": 400,
            "min_length":200,
            "temperature": 1.1,
            "num_beams": 5,
            "repetition_penalty": 1.5,
            "num_return_sequences": 4,
            "no_repeat_ngram_size": 2,
            "do_sample": True
        }
        text = generator(input_text, pad_token_id = 50256, **config_intro)
    # Return the generated text as a JSON response
    return {'generated_text': text}

templates = Jinja2Templates(directory="templates")

@app.get('/')
def get_layout(request: Request):
     return templates.TemplateResponse("index.html", {"request": request})