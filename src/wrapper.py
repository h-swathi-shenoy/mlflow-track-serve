from torch.nn import functional as F
from image_transform import ImageTransform
import requests
from PIL import Image
import mlflow


class TestWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.model = None

    def load_context(self, context):
        self.model = mlflow.pyfunc.load_model(context.artifacts['models'])

    def predict(self, context, model_input):
        print(f"Invoking predict for {model_input}")
        image = requests.get(model_input, allow_redirects=True).content
        img = Image.open(image)
        transform = ImageTransform()
        img_transform = transform(img)
        img = img_transform.unsqueeze(0)
        outs = self.model.predict(img)
        predictions = F.softmax(outs, dim=1)
        return predictions


