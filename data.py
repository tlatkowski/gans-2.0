import json
import io

caption_file = io.open('captions_val2017.json')

caption_json = json.load(caption_file)

annotations = caption_json["annotations"]
annotation_count = len(annotations)

for annotation in annotations:
    image_id = annotation["image_id"]
    caption = annotation["caption"]
    print(str(image_id) + ": " + caption)
