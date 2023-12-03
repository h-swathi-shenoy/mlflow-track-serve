from prefect import flow, task
from bing_image_downloader import downloader
from pathlib import Path
from PIL import Image
import shutil
import os

base_dir = Path(__file__).absolute().parent.parent
data_dir = base_dir.joinpath('models')
image_dir = data_dir.joinpath('cat-dogs')


@task(retries=5)
def download_images(query:str, limit:int, output_dir:Path):
    downloader.download(
        query =query,
        limit=limit,
        output_dir=output_dir,
        adult_filter_off=True, force_replace=False, timeout=60
    )


@task()
def download_validate_data(image: Path):
    download_images.fn("cat", 20, image)
    download_images.fn("dog", 20, image)
    image_dirs = os.listdir(image)
    accepted_format = [".png", ".jpg", "jpeg"]
    for img_direct in image_dirs:
        animal_dir = image_dir.joinpath(img_direct)
        for file in os.listdir(animal_dir):
            img = Image.open(animal_dir.joinpath(file))
            if img.format.lower() in accepted_format:
                pass
            else:
                print(file)
                shutil.rmtree(animal_dir.joinpath(file))
                print(f'Image need to in jpeg/png given {img.format.lower()}, removing file ')
    return


@flow()
def pipeline():
    download_validate_data(image_dir)


