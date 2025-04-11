import torch
from sentence_transformers import SentenceTransformer
from config import DEVICE


def get_embed_model():
  embedding_model = SentenceTransformer('all-MiniLM-L6-v2').to(DEVICE)
  return embedding_model

def encode_caption(captions):
  embedding_model = get_embed_model()
  encoded_captions = {}
  for image_order in captions.keys():
    caption = captions[image_order]
    encoded_captions[image_order] = {
        "embed": torch.tensor(embedding_model.encode(caption)),
        "text": caption
    }
  return encoded_captions