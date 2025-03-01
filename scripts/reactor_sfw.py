from transformers import pipeline
from PIL import Image
import logging
from scripts.reactor_logger import logger

SCORE = 0.96

logging.getLogger("transformers").setLevel(logging.ERROR)

def nsfw_image(img_path: str, model_path: str):

    with Image.open(img_path) as img:
        predict = pipeline("image-classification", model=model_path)
        result = predict(img)
        if result[0]["label"] == "nsfw" and result[0]["score"] > SCORE:
            logger.status(f"NSFW content detected, skipping...")
            return True
        return False
