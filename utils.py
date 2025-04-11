import os
import torch
import zipfile
import gdown
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

def download_file(url):
    output = None
    try:
        print(f'Downloading file from url: {url}')
        output = gdown.download(url, quiet=False, fuzzy=True)
        print(f'Downloaded to: {output}')
        print(f'Files in dir: {os.listdir(output)}')

    except Exception as error:
        print(f'Got an error: {error}')

    return output

def extract_file(path):
    if os.path.exists(path):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall("cvpr2016_flowers")
        print("Extracted successfully!")
    else:
        print("Download failed. File not found!")

def load_captions(captions_folder, images_folder):
    captions = {}
    for fname in os.listdir(captions_folder):
        with open(os.path.join(captions_folder, fname), encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
        base_name = os.path.splitext(fname)[0]
        image_file = base_name + ".jpg"
        if os.path.exists(os.path.join(images_folder, image_file)):
            captions[image_file] = lines
    return captions


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def show_grid(image):
  grid = make_grid(image.cpu(), normalize=True)
  plt.figure(figsize=(5, 5))
  plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
  plt.axis(False)
  plt.show()
