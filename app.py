import streamlit as st
import torch
import torchvision
from config import *
from model.dcgan import Generator
from model.text_encoder import get_embed_model
from utils import show_grid_pil  
from torchvision.utils import make_grid


@torch.no_grad()
def generate_image(text_embedding, generator):
    noise = torch.randn(1, NOISE_DIM, device=DEVICE)
    generated_image = generator(noise, text_embedding)
    grid = make_grid(generated_image.cpu(), normalize=True)
    return show_grid_pil(grid)  

@st.cache_resource
def load_models():
    embed_model = get_embed_model()
    generator = Generator(
        noise_dim=NOISE_DIM, 
        feature_dim=128, 
        num_channels=3,
        embedding_dim=384, 
        reduced_dim_size=256
    ).to(DEVICE)
    generator.load_state_dict(torch.load(SAVE_MODEL_PATH + 'generator_latest.pth', map_location=DEVICE))
    generator.eval()
    return embed_model, generator

def main():
    st.set_page_config(page_title="Text2Image Generator", layout="centered")
    st.title("Text2Image using BERT & DCGAN")
    st.write("Enter caption about flower to generate image.")

    # Load models
    embed_model, generator = load_models()

    # Get input from user
    input_caption = st.text_area("Caption:", value="this flower is white and purple in color, with petals that are pointed at the ends.")
    
    if st.button("Generate Image"):
        if input_caption.strip() == "":
            st.warning("Caption must not be empty.")
        else:
            with st.spinner("Generating..."):
                text_embedding = torch.tensor(embed_model.encode(input_caption)).unsqueeze(0).to(DEVICE)
                image = generate_image(text_embedding, generator)
                st.image(image, caption="Image generated from caption", use_column_width=True)

if __name__ == "__main__":
    main()