from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests


class InferlessPythonModel:
    def initialize(self):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    def infer(self, image_url):
        image = Image.open(requests.get(image_url, stream=True).raw)
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state

    def finalize(self):
        self.pipe = None
