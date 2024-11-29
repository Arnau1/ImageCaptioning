import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from transformers.image_utils import load_image
import argparse
import os
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

argparser = argparse.ArgumentParser()
argparser.add_argument("--images_path", type=str, required=True)

class Inference:
    def __init__(self, images_path):
        self.images_path = images_path
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Base")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-Instruct",
            quantization_config=quantization_config,
        ).to(DEVICE)

    def load_image(self, image_path):
        image = load_image(image_path)
        return image

    def generate_description(self, image_path):
        image = self.load_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Can you describe the image?"}
                ]
            },
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(DEVICE)
        generated_ids = self.model.generate(**inputs, max_new_tokens=50)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        # return the generated text and the image name
        return (generated_texts[0], os.path.basename(image_path))
    
    def save_to_json(self, descriptions):
        with open('descriptions.json', 'w') as f:
            json.dump(descriptions, f)

    def process_images(self):
        descriptions = []
        for image_name in os.listdir(self.images_path):
            image_path = os.path.join(self.images_path, image_name)
            if os.path.isfile(image_path):
                description = self.generate_description(image_path)
                descriptions.append(description)
        self.save_to_json(descriptions)

if __name__ == "__main__":
    args = argparser.parse_args()
    inference = Inference(args.images_path)
    inference.process_images()

