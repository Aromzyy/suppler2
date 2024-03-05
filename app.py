from flask import Flask, request, jsonify, render_template, url_for, redirect
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import io

# Assuming model1.py is in the same directory and contains necessary functions and mappings
from model1 import get_model, disease_to_index, index_to_disease

# Load your trained model as before
num_classes = len(disease_to_index)
model = get_model(num_classes)
model.load_state_dict(torch.load('DDI1_model.pth', map_location=torch.device('cpu')))
model.eval()

app = Flask(__name__)
CORS(app)

# Define the transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/', methods=['GET'])
def home():
    # GET request returns the upload form
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('home.html', error='No image provided')
        
    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_idxs = torch.topk(probabilities, 3)

        predictions = [{'class_name': index_to_disease[idx.item()], 'probability': prob.item()} for prob, idx in zip(top_probs[0], top_idxs[0])]
    
    # Render the same index.html template with predictions
    return render_template('upload.html', predictions=predictions)



@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('upload.html', error='No image provided')
            
        file = request.files['image']
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_idxs = torch.topk(probabilities, 3)

            predictions = [{'class_name': index_to_disease[idx.item()], 'probability': prob.item()} for prob, idx in zip(top_probs[0], top_idxs[0])]
        
        return render_template('upload.html', predictions=predictions)
    return render_template('upload.html')



@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/skin_conditions')
def skin_conditions():
    return render_template('skin_conditions.html')


@app.route('/FAQs')
def FAQs():
    return render_template('FAQs.html')

if __name__ == '__main__':
    app.run(debug=True)
