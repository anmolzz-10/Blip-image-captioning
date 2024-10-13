from flask import Flask, request, render_template
from transformers import BlipProcessor, BlipForConditionalGeneration, BartForConditionalGeneration, BartTokenizer
import torch
from PIL import Image
import os

app = Flask(__name__)

# Load BLIP model and processor for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load BART model and tokenizer for response generation
bart_model_name = 'facebook/bart-large-cnn'
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption

def generate_constructive_response(caption):
    prompt = f"Given the caption '{caption}', explain what this object represents."
    inputs = bart_tokenizer(prompt, return_tensors="pt")
    
    # Adjusting max_length to allow a feasible response length
    summary_ids = bart_model.generate(inputs["input_ids"], max_length=100, min_length=30, num_beams=5, early_stopping=True)
    
    response = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return response

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    # Adjust the field name based on your HTML input name
    if 'image' not in request.files:
        return "No file part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    if file:
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        # Generate caption using BLIP
        caption = generate_caption(image_path)
        
        # Generate a meaningful response using BART
        response = generate_constructive_response(caption)

        return render_template('result.html', image_filename=file.filename, caption=caption, response=response)

if __name__ == '__main__':
    app.run(debug=True)
