import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# ========== CONFIG ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "lung_model.pth"
TEST_DIR = "test_images"
OUTPUT_DIR = "gradcam_output"

# ========== CREATE OUTPUT FOLDER ==========
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== MODEL DEFINITION ==========
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = CNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ========== GRADCAM UTIL ==========
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        loss = output[:, class_idx].sum()
        loss.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-10
        return heatmap

# ========== TRANSFORM ==========
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ========== INFERENCE LOOP ==========
class_names = ['Normal', 'Cancer']
gradcam = GradCAM(model, model.conv[3])

for fname in os.listdir(TEST_DIR):
    path = os.path.join(TEST_DIR, fname)
    img = Image.open(path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probs[0][predicted].item()

    label = class_names[predicted]
    reason = f"Predicted: {label}\nConfidence: {confidence*100:.2f}%\n"

    # ========== GRADCAM ==========
    img_tensor.requires_grad = True
    heatmap = gradcam.generate(img_tensor, predicted.item())

    img_np = np.array(img.resize((128, 128)))
    if len(img_np.shape) == 2:
        img_np = np.stack([img_np]*3, axis=-1)

    heatmap = cv2.resize(heatmap, (128, 128))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

    # ========== REASONING ==========
    if label == "Cancer":
        reason += "⚠️ Regions highlighted in red show unusual tissue density indicating possible tumor location."

        # SHOW only cancer-predicted images
        plt.imshow(superimposed_img[..., ::-1])
        plt.title(reason)
        plt.axis('off')
        plt.show()

    else:
        reason += "✅ No abnormal patterns detected in lung X-ray."

    # ========== SAVE ==========
    out_path = os.path.join(OUTPUT_DIR, f"{label}_{fname}")
    cv2.imwrite(out_path, superimposed_img)
    print(f"\nImage: {fname}")
    print(reason)
