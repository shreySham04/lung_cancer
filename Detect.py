import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

# ---------------- CONFIG (auto-locate project folder) ----------------
BASE_DIR = Path(__file__).resolve().parent  # folder where this script lives
TRAIN_DIR = BASE_DIR / "lung_cancer_data" / "train"   # should contain 'cancer' & 'normal'
TEST_DIR = BASE_DIR / "test_images"                   # should contain test subfolders
OUTPUT_DIR = BASE_DIR / "gradcam_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = BASE_DIR / "lung_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 6        # increase later for better accuracy
LR = 1e-4

print("Base dir:", BASE_DIR)
print("Train dir:", TRAIN_DIR)
print("Test dir:", TEST_DIR)
print("Output dir:", OUTPUT_DIR)
print("Device:", DEVICE)

# ---------------- TRANSFORMS ----------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- LOAD TRAIN DATA ----------------
if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"Train folder not found at {TRAIN_DIR} — put 'train/cancer' and 'train/normal' there.")

train_dataset = datasets.ImageFolder(root=str(TRAIN_DIR), transform=train_transform)
class_names = train_dataset.classes  # order matters
print("Detected classes (train):", class_names)

# compute class counts and sampler to handle imbalance
targets = torch.tensor(train_dataset.targets)
class_counts = torch.bincount(targets)
print("Class counts:", class_counts.tolist())

# avoid division by zero:
class_counts = class_counts.float()
class_weights = (1.0 / (class_counts + 1e-8))
sample_weights = class_weights[targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)

# ---------------- MODEL (ResNet18) ----------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(DEVICE)

# loss with class weighting (helpful for imbalance)
cw = class_weights.to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=cw)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------------- TRAIN IF NEEDED ----------------
if MODEL_PATH.exists():
    print("Loading existing model:", MODEL_PATH)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=DEVICE))
else:
    print("Training model...")
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), str(MODEL_PATH))
    print("Saved model to", MODEL_PATH)
# GRAD-CAM HELPER 
# target conv layer: last conv of resnet layer4
target_layer = model.layer4[-1].conv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        # hooks
        target_layer.register_forward_hook(self._forward_hook)
        # use full backward hook
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        # output shape: (N, C, H, W)
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output[0] shape: (N, C, H, W)
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: 1xCxHxW on DEVICE
        class_idx: int (target class). If None, uses predicted class.
        returns: heatmap HxW (float 0..1)
        """
        self.model.zero_grad()
        out = self.model(input_tensor)            # forward
        if class_idx is None:
            class_idx = int(out.argmax(dim=1).item())
        score = out[0, class_idx]
        score.backward(retain_graph=False)

        gradients = self.gradients.cpu().numpy()[0]   # (C, H, W)
        activations = self.activations.cpu().numpy()[0]  # (C, H, W)

        weights = np.mean(gradients, axis=(1, 2))     # (C,)
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # (H, W)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        if cam.max() != 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        return cam

gradcam = GradCAM(model, target_layer)

# ---------------- COLLECT TEST IMAGE PATHS (recursively) ----------------
if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test directory not found: {TEST_DIR}. Put test images under test_images/ subfolders.")

test_images_list = []
for root, _, files in os.walk(TEST_DIR):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            test_images_list.append(Path(root) / f)

if len(test_images_list) == 0:
    raise FileNotFoundError(f"No test images found under {TEST_DIR} (recursively).")

print(f"Found {len(test_images_list)} test images. Starting inference + Grad-CAM...")

# ---------------- INFERENCE + SAVE ----------------
softmax = nn.Softmax(dim=1)
cancer_index = None
# find which index corresponds to 'cancer' (case-insensitive) in class_names
for idx, name in enumerate(class_names):
    if "cancer" in name.lower():
        cancer_index = idx
        break

if cancer_index is None:
    # fallback: assume index 0 = cancer if class name not found
    cancer_index = 0
    print("Warning: could not find 'cancer' in class names; assuming index 0 is cancer:", class_names)

# create subfolders for outputs
(OUTPUT_DIR / "cancer").mkdir(exist_ok=True)
(OUTPUT_DIR / "normal").mkdir(exist_ok=True)

model.eval()
for img_path in test_images_list:
    try:
        pil = Image.open(img_path).convert("RGB")
    except Exception as e:
        print("Skipping unreadable:", img_path, e)
        continue

    inp = test_transform(pil).unsqueeze(0).to(DEVICE)  # 1x3xHxW

    # predict
    with torch.no_grad():
        out = model(inp)
        probs = softmax(out)
        conf_val, pred_idx = torch.max(probs, 1)
        pred_idx = int(pred_idx.item())
        conf_val = float(conf_val.item())

    pred_label = class_names[pred_idx]
    fname = img_path.name

    if pred_idx == cancer_index:
        # generate Grad-CAM
        cam = gradcam.generate(inp, pred_idx)  # 0..1
        heatmap = np.uint8(255 * cam)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        orig = np.array(pil.resize((IMG_SIZE, IMG_SIZE)))  # RGB uint8 HxWx3
        overlay = cv2.addWeighted(orig, 0.6, heatmap_color[..., ::-1], 0.4, 0)  # heatmap_color is BGR

        # side-by-side (Original | Overlay)
        combined = np.hstack((orig, overlay))
        out_fname = f"CANCER_{conf_val*100:.1f}_{fname}"
        out_path = OUTPUT_DIR / "cancer" / out_fname
        # cv2 expects BGR for saving:
        cv2.imwrite(str(out_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        reason = f"⚠️ Cancer predicted — Confidence: {conf_val*100:.2f}%\nRed areas = model focus."
        print(f"[CANCER] {fname} -> {pred_label} ({conf_val*100:.2f}%)  saved: {out_path.name}")

        # display
        plt.figure(figsize=(10,5))
        plt.imshow(combined)
        plt.title(reason)
        plt.axis("off")
        plt.close() #Replace it to plt.show whenever you want image output with gradcam output

    else:
        # Normal: save original only
        orig = np.array(pil.resize((IMG_SIZE, IMG_SIZE)))
        out_fname = f"NORMAL_{conf_val*100:.1f}_{fname}"
        out_path = OUTPUT_DIR / "normal" / out_fname
        cv2.imwrite(str(out_path), cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))

        reason = f"✅ Normal — Confidence: {conf_val*100:.2f}%\nNo suspicious areas highlighted."
        print(f"[NORMAL] {fname} -> {pred_label} ({conf_val*100:.2f}%)  saved: {out_path.name}")

        plt.figure(figsize=(6,6))
        plt.imshow(orig)
        plt.title(reason)
        plt.axis("off")
        plt.close() ##Replace it to plt.show whenever you want image output with gradcam output


print("All done. Grad-CAM outputs saved in:", OUTPUT_DIR)

# ===================== EVALUATION =====================
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

print("\nEvaluating model on test set...")

# Put model in eval mode
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Print results
print("Accuracy:", accuracy_score(all_labels, all_preds))
print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=classes))
print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))

