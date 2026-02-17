import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(path):
    img = Image.open(path)
    img = img.resize((512, 512))
    img = np.array(img) / 255.0
    img = img.astype(np.float32)
    img = img[np.newaxis, ...]
    return tf.convert_to_tensor(img)

def save_image(tensor, filename):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    Image.fromarray(tensor).save(filename)

def main():
    print("Loading Neural Style Transfer model...")

    model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

    content_path = input("Enter path of content image: ")
    style_path = input("Enter path of style image: ")

    content_image = load_image(content_path)
    style_image = load_image(style_path)

    print("Applying style...")

    stylized_image = model(content_image, style_image)[0]

    save_image(stylized_image, "stylized_output.png")

    print("Stylized image saved as stylized_output.png")

if __name__ == "__main__":
    main()
