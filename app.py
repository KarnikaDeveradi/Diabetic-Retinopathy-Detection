from flask import Flask, request, render_template, redirect, url_for
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define EfficientNetB5 Model
class EfficientNetB5(nn.Module):
    def __init__(self, num_classes=5):
        super(EfficientNetB5, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5')
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)
        self.features = None
        self.gradients = None
        self.model._conv_head.register_forward_hook(self.save_features)
        self.model._conv_head.register_full_backward_hook(self.save_gradients)

    def save_features(self, module, input, output):
        self.features = output

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def forward(self, x):
        return self.model(x)

# Load model
model = EfficientNetB5(num_classes=5).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Class names
class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# Preprocessing function
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Unable to load image")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = min(width, height) // 2 - 5
    cv2.circle(mask, center, radius, 255, -1)
    image = cv2.bitwise_and(image, image, mask=mask)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

# Grad-CAM++ implementation
def grad_cam_plus_plus(model, image, target_class):
    model.eval()
    image = image.to(device)
    image.requires_grad = True
    
    output = model(image)
    score = output[0, target_class]
    
    model.zero_grad()
    score.backward()
    
    gradients = model.gradients
    features = model.features
    
    alpha_num = gradients ** 2
    alpha_denom = 2 * (gradients ** 2) + torch.sum(features * (gradients ** 3), dim=(2, 3), keepdim=True)
    alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))
    alpha = alpha_num / alpha_denom
    
    weights = torch.nn.functional.relu(torch.sum(alpha * torch.nn.functional.relu(gradients), dim=(2, 3), keepdim=True))
    heatmap = torch.sum(weights * features, dim=1).squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1
    return heatmap

# Overlay heatmap on image (Fixed)
def overlay_heatmap(input_tensor, heatmap):
    # Detach the tensor from the computation graph before converting to NumPy
    img = input_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    img = img.astype(np.uint8)
    
    heatmap = cv2.resize(heatmap, (380, 380))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed = heatmap * 0.4 + img
    return superimposed.astype(np.uint8)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    if not file:
        return "No file uploaded", 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    
    try:
        # Preprocess and predict
        input_tensor = preprocess_image(file_path)
        input_tensor = input_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1)[0, pred_class].item()
        
        # Grad-CAM++
        heatmap = grad_cam_plus_plus(model, input_tensor, pred_class)
        superimposed = overlay_heatmap(input_tensor, heatmap)
        
        # Convert to base64
        original_img = cv2.imread(file_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        _, orig_buffer = cv2.imencode(".png", original_img)
        orig_b64 = base64.b64encode(orig_buffer).decode("utf-8")
        
        _, super_buffer = cv2.imencode(".png", superimposed)
        super_b64 = base64.b64encode(super_buffer).decode("utf-8")
        
        # Clean up
        os.remove(file_path)
        
        return render_template("result.html",
                              diagnosis=class_names[pred_class],
                              confidence=f"{(confidence * 100):.1f}%",
                              original_image=orig_b64,
                              heatmap_image=super_b64,
                              severity=pred_class)
    except Exception as e:
        os.remove(file_path)
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)