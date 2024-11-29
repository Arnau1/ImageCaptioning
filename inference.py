import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from transformers.image_utils import load_image
import argparse
import os
import json 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

argparser = argparse.ArgumentParser()
argparser.add_argument("--image_path", type=str, required=True)
argparser.add_argument("--ground_truth", type=str, required=False)

class Inference:
    def __init__(self, image_path, ground_truth = None):
        self.image_path = image_path
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Base")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-Instruct",
            quantization_config=quantization_config,
        )

    def load_image(self):
        image = load_image(self.image_path)
        return image

    def generate_description(self):
        image = self.load_image()
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
        return generated_texts[0]
    
    def save_to_json(self, description):
        with open('description.json', 'w') as f:
            json.dump(description, f)

if __name__ == "__main__":
    args = argparser.parse_args()
    inference = Inference(args.image_path, args.ground_truth)
    description = inference.generate_description()
    inference.save_to_json(description)
    print(description)
