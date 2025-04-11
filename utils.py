import os
import torch
import zipfile
import gdown
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
import torch
from config import DEVICE

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
  image_files = os.listdir(images_folder)

  for image_file in image_files:
    image_name = image_file.split('.')[0]
    caption_file = os.path.join(captions_folder, image_name + '.txt')
    with open(caption_file, 'r') as f:
      caption = f.readlines()[0].strip()
      if image_name not in captions:
        captions[image_name] = caption

  return captions

def save_model(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path, 'generator.pth')
    torch.save(model.state_dict(), file_path)
    print(f'Model saved to {file_path}')

def load_model(model, path):
    if not os.path.exists(path):
        print("Model file not found. Please train the model first.")
        return None
    
    print(f"Loading model from {path}...")
    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    return model
    
def show_grid(image):
  grid = make_grid(image.cpu(), normalize=True)
  plt.figure(figsize=(5, 5))
  plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
  plt.axis('off')
  plt.show()
