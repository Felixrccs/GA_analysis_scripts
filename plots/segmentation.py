import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, feature, segmentation
from sklearn.cluster import KMeans

# Load image
img = io.imread("crossover_plot.png")

# Convert to grayscale if needed
if img.ndim == 3:
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img_gray = color.rgb2gray(img)
else:
    img_gray = img

# Smooth slightly to reduce atomic-level noise
img_smooth = filters.gaussian(img_gray, sigma=2.0)

# --- Compute structure tensor ---
Axx, Axy, Ayy = feature.structure_tensor(img_smooth, sigma=3)
orientation = 0.5 * np.arctan2(2 * Axy, Axx - Ayy)  # local orientation angle

# Normalize to [0,1]
orientation_norm = (orientation - orientation.min()) / (orientation.max() - orientation.min())

# Flatten for clustering
pixels = orientation_norm.reshape(-1, 1)

# --- Cluster into 3 domains (CuO, CuOH, CuH) ---
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(pixels)
segmented = labels.reshape(img_gray.shape)

# --- Display ---
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(img_gray, cmap='gray')
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(orientation_norm, cmap='hsv')
plt.title("Local orientation")

plt.subplot(1, 3, 3)
plt.imshow(segmented, cmap='tab10')
plt.title("Segmented domains (KMeans)")
plt.tight_layout()
plt.show()
