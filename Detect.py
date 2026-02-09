# Detect.py
# Lung cancer detection + focused Grad-CAM + grey lung + red cancer 3D visualization

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

import open3d as o3d

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "lung_cancer_data" / "train"
TEST_DIR  = BASE_DIR / "test_images"
OUTPUT_DIR = BASE_DIR / "gradcam_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = BASE_DIR / "lung_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", DEVICE)

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 6
LR = 1e-4

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
    raise FileNotFoundError(TRAIN_DIR)

train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transform)
class_names = train_dataset.classes
print("Classes:", class_names)

targets = torch.tensor(train_dataset.targets)
class_counts = torch.bincount(targets).float()
class_weights = 1.0 / (class_counts + 1e-8)

sample_weights = class_weights[targets]
sampler = WeightedRandomSampler(sample_weights,
                                num_samples=len(sample_weights),
                                replacement=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler
)

# ---------------- MODEL ----------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------------- TRAIN OR LOAD ----------------
if MODEL_PATH.exists():
    print("Loading model...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    print("Training model...")
    model.train()
    for e in range(EPOCHS):
        loss_sum = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        print(f"Epoch {e+1}/{EPOCHS}  Loss: {loss_sum/len(train_loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved.")

# ---------------- GRAD-CAM ----------------
target_layer = model.layer4[-1].conv2

class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.activations = None
        self.gradients = None
        layer.register_forward_hook(self._fw)
        layer.register_full_backward_hook(self._bw)

    def _fw(self, m, i, o):
        self.activations = o.detach()

    def _bw(self, m, gi, go):
        self.gradients = go[0].detach()

    def generate(self, x, class_idx):
        self.model.zero_grad()
        out = self.model(x)
        score = out[0, class_idx]
        score.backward()

        acts = self.activations[0].cpu().numpy()
        grads = self.gradients[0].cpu().numpy()

        w = grads.mean(axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)

        for i, wi in enumerate(w):
            cam += wi * acts[i]

        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

        return cam

gradcam = GradCAM(model, target_layer)

# ---------------- 3D focused visualization ----------------
def gradcam_to_3d(rgb, cam, mask):

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    depth = cv2.GaussianBlur(gray, (21, 21), 0)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth = 1.0 - depth

    # keep only strongest activations
    thr = np.percentile(cam, 92)

    h, w = cam.shape
    pts, cols = [], []

    step = 4   # larger step -> more transparent look

    for y in range(0, h, step):
        for x in range(0, w, step):

            if mask[y, x] < 0.5:
                continue

            z = depth[y, x] * 80.0
            pts.append([x, y, z])

            if cam[y, x] >= thr:
                # cancer
                col = np.array([1.0, 0.0, 0.0])
            else:
                # lung (grey)
                col = np.array([0.7, 0.7, 0.7])

            cols.append(col)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(cols))

    return pcd

# ---------------- COLLECT TEST IMAGES ----------------
if not TEST_DIR.exists():
    raise FileNotFoundError(TEST_DIR)

test_images = []
for r, _, f in os.walk(TEST_DIR):
    for file in f:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            test_images.append(Path(r) / file)

print("Test images:", len(test_images))

softmax = nn.Softmax(dim=1)

cancer_index = 0
for i, n in enumerate(class_names):
    if "cancer" in n.lower():
        cancer_index = i

(OUTPUT_DIR / "cancer").mkdir(exist_ok=True)
(OUTPUT_DIR / "normal").mkdir(exist_ok=True)

# ---------------- INFERENCE ----------------
model.eval()

for img_path in test_images:

    pil = Image.open(img_path).convert("RGB")
    inp = test_transform(pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(inp)
        prob = softmax(out)
        conf, pred = torch.max(prob, 1)

    pred = int(pred.item())
    conf = float(conf.item())

    resized = np.array(pil.resize((IMG_SIZE, IMG_SIZE)))

    if pred == cancer_index:

        cam = gradcam.generate(inp, pred)

        # focus sharpening
        cam = cv2.GaussianBlur(cam, (11, 11), 0)
        cam = cam ** 2

        # 2D overlay for reference
        heat = np.uint8(cam * 255)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(
            resized, 0.6,
            heat_color[..., ::-1], 0.4, 0
        )
        combined = np.hstack((resized, overlay))

        out_path = OUTPUT_DIR / "cancer" / f"CANCER_{conf*100:.1f}_{img_path.name}"
        cv2.imwrite(str(out_path),
                    cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        # rough lung mask
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.medianBlur(mask, 11)
        mask = mask / 255.0

        # 3D
        pcd = gradcam_to_3d(resized, cam, mask)

        ply_path = OUTPUT_DIR / "cancer" / f"3D_{img_path.stem}.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)

        print(f"[CANCER] {img_path.name}  {conf*100:.2f}%")

        plt.figure(figsize=(10, 5))
        plt.imshow(combined)
        plt.title(f"Cancer {conf*100:.2f}% (right = Grad-CAM)")
        plt.axis("off")
        plt.show()

        o3d.visualization.draw_geometries([pcd])

    else:

        out_path = OUTPUT_DIR / "normal" / f"NORMAL_{conf*100:.1f}_{img_path.name}"
        cv2.imwrite(str(out_path),
                    cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

        print(f"[NORMAL] {img_path.name}  {conf*100:.2f}%")

        plt.figure(figsize=(5, 5))
        plt.imshow(resized)
        plt.title(f"Normal {conf*100:.2f}%")
        plt.axis("off")
        plt.show()

print("\nDone. Grey lung + red cancer 3D models saved in:", OUTPUT_DIR)
