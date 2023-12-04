from pathlib import Path
import requests
import os
from torch.nn import functional as F
from image_transform import ImageTransform
from PIL import Image


base_dir = Path(__file__).absolute().parent.parent
data_dir = base_dir.joinpath('data')
test_folder = data_dir.joinpath('test')
image = os.listdir(test_folder)[0]

if __name__ == "__main__":
    endpoint = "http://localhost:7777/invocations"
    data = {
        'instances': str(test_folder.joinpath(image))
    }

    response = requests.post(endpoint, json=data)
    print(response.json())
    predictions = eval(response.text)["predictions"]
    print("This image is {:.2f}% cat and {:.2f}% dog.".format(100 * float(predictions[0][0]),
                                                              100 * float(predictions[0][1])))