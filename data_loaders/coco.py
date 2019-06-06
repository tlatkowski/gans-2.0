import io
import json

from tensorflow.python.keras import preprocessing

CAPTIONS_DIR = '../annotations/captions_val2017.json'
IMAGES_DIR = '../val2017'


def load_json_captions(path: str):
    caption_file = io.open(path)
    caption_json = json.load(caption_file)
    return caption_json


captions_json = load_json_captions(CAPTIONS_DIR)

annotations = captions_json["annotations"]
annotation_count = len(annotations)
tokenizer = preprocessing.text.Tokenizer(num_words=10)

for annotation in annotations:
    image_id = annotation["image_id"]
    caption = annotation["caption"]
    tokenizer.fit_on_texts(caption)
    vectorized_caption = tokenizer.texts_to_sequences(caption)
    print(str(image_id) + ": " + caption)
    
    image_fn = "{:0>12}.jpg".format(image_id)
    print(image_fn)
