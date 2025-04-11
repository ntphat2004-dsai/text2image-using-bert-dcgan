import torch
from sentence_transformers import SentenceTransformer
from config import DEVICE


def get_embed_model():
  embedding_model = None
  embedding_model = SentenceTransformer('all-MiniLM-L6-v2').to(DEVICE)
  return embedding_model

def encode_caption(captions):
  """
  Encodes a dictionary of image captions into embeddings using a pre-trained embedding model.

  Args:
    captions (dict): A dictionary where keys are image identifiers (e.g., image order) 
             and values are the corresponding captions (strings).

  Returns:
    dict: A dictionary where keys are the same image identifiers as the input, and values 
        are dictionaries containing:
        - "embed": A PyTorch tensor representing the embedding of the caption.
        - "text": The original caption string.

  Notes:
    - The function relies on a helper function `get_embed_model()` to retrieve the 
      pre-trained embedding model.
    - The embeddings are computed using the `encode` method of the embedding model.
    - Ensure that the `torch` library is imported and available in the environment.
  """
  embedding_model = get_embed_model()
  encoded_captions = {}
  for image_order in captions.keys():
    caption = captions[image_order]
    encoded_captions[image_order] = {
        "embed": torch.tensor(embedding_model.encode(caption)),
        "text": caption
    }
  return encoded_captions
