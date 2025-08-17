import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# -------- CONFIG --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = "lung_cancer_data"   # Should have subfolders: Normal, Cancer
TEST_DIR = "test_images"         # Also should have subfolders: Normal, Cancer
MODEL_PATH = "lung_model.pth"
OUTPUT_DIR = "gradcam_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001

# -------- MODEL --------
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

# -------- DATA TRANSFORMS --------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# -------- TRAIN DATA --------
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------- LOSS & OPTIMIZER --------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------- TRAIN LOOP --------
print("Starting training...")
model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# -------- SAVE MODEL --------
torch.save(model.state_dict(), MODEL_PATH)
print(f"Training complete. Model saved as {MODEL_PATH}")

# -------- GRADCAM HOOKS --------
gradients = None
activations = None

def save_gradients_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def save_activations_hook(module, input, output):
    global activations
    activations = output

# Hook to last Conv2d layer
model.conv[3].register_forward_hook(save_activations_hook)
model.conv[3].register_full_backward_hook(save_gradients_hook)

# -------- TEST LOOP --------
if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

print("\nTesting images...")
model.eval()
class_names = ['Normal', 'Cancer']

for root, _, files in os.walk(TEST_DIR):
    for fname in files:
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(root, fname)
        image = Image.open(path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Prediction
        output = model(img_tensor)
        pred_class = torch.argmax(output, dim=1)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence = probs[0][pred_class].item()
        label = class_names[pred_class]

        print(f"Image: {fname} --> Predicted: {label} ({confidence*100:.2f}%)")

        # Grad-CAM
        model.zero_grad()
        output[0, pred_class].backward()

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activation_map = activations[0].detach()

        for i in range(len(pooled_gradients)):
            activation_map[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activation_map, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= heatmap.max() + 1e-10

        # Overlay heatmap on original image
        img_np = np.array(image.resize((128, 128)))
        if len(img_np.shape) == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        heatmap_resized = cv2.resize(heatmap, (128, 128))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

        # Reasoning
        if label == "Cancer":
            reason = f"⚠️ Cancer detected\nConfidence: {confidence*100:.2f}%\nRed zones = suspicious tissue."
        else:
            reason = f"✅ Normal lungs\nConfidence: {confidence*100:.2f}%\nNo abnormal high-density regions."

        # Save Grad-CAM output
        out_path = os.path.join(OUTPUT_DIR, f"{label}_{fname}")
        cv2.imwrite(out_path, overlay)

        # Show image
        plt.imshow(overlay[..., ::-1])
        plt.title(reason)
        plt.axis('off')
        plt.show()
