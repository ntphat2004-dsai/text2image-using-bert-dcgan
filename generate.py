import torch
import torchvision
from config import *
from model.dcgan import Generator
from model.text_encoder import get_embed_model
from utils import show_grid
import os

def generate_image(input_text, generator):
    generator.eval()
    embed_model = get_embed_model()
    text_embedding = torch.tensor(embed_model.encode(input_text))
    text_embedding = text_embedding.unsqueeze(0).to(DEVICE)
    noise = torch.randn(1, NOISE_DIM, device=DEVICE)

    with torch.inference_mode():
        generated_image = generator(noise, text_embedding)

    grid = torchvision.utils.make_grid(generated_image.cpu(), normalize=True)
    show_grid(grid)


def main():
    generator = Generator(
        noise_dim=NOISE_DIM, 
        feature_dim=128, 
        num_channels=3,
        embedding_dim=384, 
        reduced_dim_size=256
    ).to(DEVICE)

    path = os.path.join(SAVE_MODEL_PATH, 'generator.pth')

    generator = load_model(generator, path)

    input_text = input('Enter a caption: ') # Example: "this flower is white and purple in color, with petals that are pointed at the ends."
    generate_image(input_text)
    
if __name__ == "__main__":
    main()