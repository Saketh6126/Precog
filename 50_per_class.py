import numpy as np
import gzip
import os

# Paths
data_dir = 'Data/Raw/MNIST/raw'
img_file = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
lbl_file = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')

# Load images and labels
with gzip.open(lbl_file, 'rb') as f:
    labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

with gzip.open(img_file, 'rb') as f:
    images = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    images = images.reshape(-1, 28, 28)

# Save 50 images per class in separate files (no labels)
for digit in range(10):
    indices = np.where(labels == digit)[0][:50]
    digit_images = images[indices]
    np.save(f'mnist_class_{digit}.npy', digit_images)
    print(f"Saved mnist_class_{digit}.npy with shape {digit_images.shape}")

# Output: 10 files, each (50, 28, 28)   