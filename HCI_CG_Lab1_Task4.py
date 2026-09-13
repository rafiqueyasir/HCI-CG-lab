# Task 4 — Spatial Downsampling & Pixelation via Striding

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Make sure sample.jpg is in the same folder as this Python file
image = Image.open("sample.jpg").convert("RGB")
img = np.array(image)

# Downsample by taking every 8th pixel
downsampled = img[::8, ::8, :]

# Re-expand using np.repeat()
expanded = np.repeat(np.repeat(downsampled, 8, axis=0), 8, axis=1)

# Crop to original dimensions
expanded = expanded[:img.shape[0], :img.shape[1], :]

# Display original, downsampled and pixelated images
plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(downsampled)
plt.title("Downsampled (Every 8th Pixel)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(expanded)
plt.title("Pixelated / Re-expanded")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Original Shape:", img.shape)
print("Original Memory:", img.nbytes, "bytes")

print("Downsampled Shape:", downsampled.shape)
print("Downsampled Memory:", downsampled.nbytes, "bytes")

print("Re-expanded Shape:", expanded.shape)

dimension_reduction = (1 - (downsampled.shape[0] / img.shape[0])) * 100
memory_savings = (1 - (downsampled.nbytes / img.nbytes)) * 100

print("Dimension Reduction: {:.2f}%".format(dimension_reduction))
print("Memory Savings: {:.2f}%".format(memory_savings))
