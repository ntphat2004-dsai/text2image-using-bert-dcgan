import torch
import torchvision
from config import *
from model.dcgan import Generator
from model.text_encoder import get_embed_model
from utils import show_grid
import glob
import os

def generate_image(text_embedding, generator):
    noise = torch.randn(1, NOISE_DIM, device=DEVICE)

    with torch.inference_mode():
        generated_image = generator(noise, text_embedding)

    grid = torchvision.utils.make_grid(generated_image.cpu(), normalize=True)
    show_grid(grid)


def main():
    embed_model = get_embed_model()

    generator = Generator(
        noise_dim=NOISE_DIM, 
        feature_dim=128, 
        num_channels=3,
        embedding_dim=384, 
        reduced_dim_size=256
    ).to(DEVICE)

    path = os.path.join(SAVE_MODEL_PATH, 'generator.pth')
    if not os.path.exists(path):
        print("Model file not found. Please train the model first.")
        return
    print(f"Loading model from {path}...")
    state_dict = torch.load(path, map_location=DEVICE)
    generator.load_state_dict(state_dict)

    input_caption = input('Enter a caption: ') # Example: "this flower is white and purple in color, with petals that are pointed at the ends."
    text_embedding = torch.tensor(embed_model.encode(input_caption))
    text_embedding = text_embedding.unsqueeze(0).to(DEVICE)

    generate_image(text_embedding, generator)
    
if __name__ == "__main__":
    main()