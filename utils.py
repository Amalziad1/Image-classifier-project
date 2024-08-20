import numpy as np
from PIL import Image
import tensorflow as tf

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

def plot_predictions(image_path, model, top_k):
    image = Image.open(image_path)    
    probs, class_indices = predict(image_path, model, top_k)
    class_labels = [class_names[str(index)] for index in class_indices]
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.barh(class_labels, probs, color='blue')
    plt.xlabel('Probability')
    plt.title('Top {} Predictions'.format(top_k))
    
    plt.tight_layout()
    plt.show()