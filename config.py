import torch

IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_EPOCHS = 1
NOISE_DIM = 100
REAL_LABEL = 0.9
FAKE_LABEL = 0.1
SAVE_MODEL_PATH = "./saved_models/"
SAVE_IMAGE_PATH = "./saved_images/"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_URL = "https://drive.google.com/uc?id=1JJjMiNieTz7xYs6UeVqd02M3DW4fnEfU"
CAPTIONS_PATH = "./cvpr2016_flowers/content/cvpr2016_flowers/captions"
IMAGES_PATH = "./cvpr2016_flowers/content/cvpr2016_flowers/images"