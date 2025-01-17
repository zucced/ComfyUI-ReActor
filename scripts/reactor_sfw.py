from transformers import pipeline
from PIL import Image
import logging

SCORE = 0.85

logging.getLogger('transformers').setLevel(logging.ERROR)

img_path = "test.jpg"
model_path = "models/vit-base-nsfw-detector"

def nsfw_image(img_path: str, model_path: str):
    img = Image.open(img_path)
    predict = pipeline("image-classification", model=model_path)
    result = predict(img)
    return True if result[0]["score"] > SCORE else False
