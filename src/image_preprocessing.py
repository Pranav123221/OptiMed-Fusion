
import numpy as np
from PIL import Image

def preprocess_scan(uploaded_file):
    """
    Convert an uploaded image into a 32x32 grayscale
    float32 array for the CNN.
    """

    image = Image.open(uploaded_file).convert("L")
    image = image.resize((32, 32))

    image_array = np.asarray(image, dtype=np.float32) / 255.0

    image_array = np.expand_dims(image_array, axis=-1)

    if image_array.shape != (32, 32, 1):
        raise ValueError("Invalid scan dimensions.")

    if not np.isfinite(image_array).all():
        raise ValueError("Invalid image pixel values.")

    return image_array