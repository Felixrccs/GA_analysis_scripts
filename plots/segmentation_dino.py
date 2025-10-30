import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from sklearn.cluster import KMeans
from timm import create_model
from skimage import morphology, color, segmentation

# ------------------- CONFIGURATION -------------------
# Use Apple GPU (MPS) if available, otherwise fallback to CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running on device: {device}")

# DINO model (faster and lighter than DINOv2)
model = create_model("vit_small_patch16_224.dino", pretrained=True).to(device)
model.eval()

# Load image
img = Image.open("crossover_plot.png").convert("RGB")

# Optionally downscale for faster processing (you can comment this out)
#img.thumbnail((1024, 1024))   # reduce resolution for speed
W, H = img.size

# Segmentation parameters
tile = 224       # model input size
stride = 16      # overlap between tiles (lower = finer but slower)
n_clusters = 3   # CuO / CuOH / CuH domains

# Normalization and preprocessing for DINO
transform = transforms.Compose([
    transforms.Resize((tile, tile)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# ------------------- FEATURE EXTRACTION -------------------
print("Extracting DINO embeddings...")

feats_all, coords = [], []

for y in range(0, H - tile + 1, stride):
    for x in range(0, W - tile + 1, stride):
        crop = img.crop((x, y, x + tile, y + tile))
        x_t = transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            f = model.forward_features(x_t)

        # Handle different timm model output structures
        if isinstance(f, dict):
            f = f.get("x_norm_patchtokens", list(f.values())[0])
        if f.shape[1] == 197:  # drop CLS token
            f = f[:, 1:, :]

        # Average embeddings of all patches within the tile
        feats_all.append(f.squeeze(0).cpu().numpy().mean(axis=0))
        coords.append((x + tile // 2, y + tile // 2))

feats_all = np.array(feats_all)
coords = np.array(coords)
print(f"Extracted {feats_all.shape[0]} feature vectors of dimension {feats_all.shape[1]}")

# ------------------- CLUSTERING -------------------
print("Clustering into domains...")
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(feats_all)

# ------------------- RECONSTRUCT SEGMENTATION -------------------
print("Reconstructing segmentation map...")
seg_map = np.zeros((H, W), dtype=np.uint8)

for (cx, cy), lab in zip(coords, labels):
    y0, y1 = max(cy - stride, 0), min(cy + stride, H)
    x0, x1 = max(cx - stride, 0), min(cx + stride, W)
    seg_map[y0:y1, x0:x1] = lab

# Morphological smoothing
seg_map = morphology.opening(seg_map, morphology.disk(3))
seg_map = morphology.closing(seg_map, morphology.disk(5))

# Overlay segmentation on original image
overlay = color.label2rgb(seg_map, image=np.asarray(img), bg_label=0, alpha=0.4)

# ------------------- DISPLAY RESULTS -------------------
plt.figure(figsize=(14, 6))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(seg_map, cmap="tab10")
plt.title("DINO Segmentation (MPS GPU)")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("Overlay on Original")
plt.tight_layout()
plt.show()

# ------------------- SAVE RESULTS -------------------
from PIL import Image as PILImage
seg_img = PILImage.fromarray(seg_map)
seg_img.save("segmentation_dino_mac.png")
print("✅ Segmentation saved as segmentation_dino_mac.png")
