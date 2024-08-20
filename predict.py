import argparse
import json
import numpy as np
from keras.models import load_model
from utils import process_image, predict

def main():
    parser = argparse.ArgumentParser(description='Predict flower class from an image.')
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    parser.add_argument('model_path', type=str, help='Path to the saved model.')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes.')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names.')
    args = parser.parse_args()

    model = load_model(args.model_path)

    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
    else:
        class_names = None

    probs, class_indices = predict(args.image_path, model, args.top_k)
    if class_names:
        class_labels = [class_names[str(index)] for index in class_indices]
    else:
        class_labels = class_indices

    print("Top {} predictions:".format(args.top_k))
    for prob, label in zip(probs, class_labels):
        print(f"{label}: {prob:.4f}")

if __name__ == "__main__":
    main()
