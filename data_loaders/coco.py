import io
import json
import os

import numpy as np
from tensorflow.python.keras import preprocessing

CAPTIONS_DIR = '../annotations/captions_val2017.json'
IMAGES_DIR = '../val2017'


def load_json_captions(path: str):
    caption_file = io.open(path)
    caption_json = json.load(caption_file)
    return caption_json


def load_data(captions_path: str, images_path: str):
    captions_json = load_json_captions(captions_path)
    annotations = captions_json["annotations"]
    
    images_paths = []
    captions_texts = []
    for annotation in annotations:
        image_id = annotation["image_id"]
        caption_txt = annotation["caption"]
        image_fn = os.path.join(images_path, "{:0>12}.jpg".format(image_id))
        
        images_paths.append(image_fn)
        captions_texts.append(caption_txt)
    
    images_paths = np.array(images_paths)
    captions_texts = np.array(captions_texts)
    
    return images_paths, captions_texts


def load_vactorized_data(captions_path: str, images_path: str):
    captions_json = load_json_captions(captions_path)
    
    annotations = captions_json["annotations"]
    tokenizer = preprocessing.text.Tokenizer(num_words=10)
    
    images_paths = []
    captions_texts = []
    for annotation in annotations:
        image_id = annotation["image_id"]
        caption_txt = annotation["caption"]
        
        tokenizer.fit_on_texts(caption_txt)
        vectorized_caption = tokenizer.texts_to_sequences(caption_txt)
        image_fn = os.path.join(images_path, "{:0>12}.jpg".format(image_id))
        images_paths.append(image_fn)
        captions_texts.append(vectorized_caption)
    
    images_paths = np.array(images_paths)
    captions_texts = np.array(captions_texts)
    
    return images_paths, captions_texts


load_data(CAPTIONS_DIR, IMAGES_DIR)
load_vactorized_data(CAPTIONS_DIR, IMAGES_DIR)
