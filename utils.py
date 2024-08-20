import numpy as np
from PIL import Image
import tensorflow as tf

def process_image(image_np):
    image_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)
    image_resized = tf.image.resize(image_tensor, (224, 224))
    image_normalized = image_resized / 255.0
    image_processed = image_normalized.numpy()
    return image_processed

def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image_np = np.asarray(image)
    image_processed = process_image(image_np)
    image_processed = np.expand_dims(image_processed, axis=0)
    predictions = model.predict(image_processed)
    top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]
    top_k_probs = predictions[0][top_k_indices]
    top_k_classes = [str(index) for index in top_k_indices]
    
    return top_k_probs, top_k_classes

