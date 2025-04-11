import torch
import torch.nn as nn

# noise_dim=100, feature_dim=128, num_channels=3, embedding_dim=768, reduced_dim_size=256
class Generator(nn.Module):
  def __init__(self, noise_dim, feature_dim, num_channels, embedding_dim, reduced_dim_size):
    super(Generator, self).__init__()
    self.reduced_dim_size = reduced_dim_size

    # Text encoder to reduce the dimensionality of the text embedding
    self.text_encoder = nn.Sequential(
        nn.Linear(in_features=embedding_dim, out_features=reduced_dim_size),
        nn.BatchNorm1d(num_features=reduced_dim_size),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
    )

    # Main generator network to generate images from noise and text embeddings
    self.main = nn.Sequential(
        nn.ConvTranspose2d(noise_dim + reduced_dim_size, feature_dim*8, 4, 1, 0, bias=False),  # First deconvolution layer
        nn.BatchNorm2d(feature_dim*8),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),

        nn.ConvTranspose2d(feature_dim*8, feature_dim*4, 4, 2, 1, bias=False),  # Upsampling
        nn.BatchNorm2d(feature_dim*4),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(feature_dim*4, feature_dim*2, 4, 2, 1, bias=False),  # Upsampling
        nn.BatchNorm2d(feature_dim*2),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(feature_dim*2, feature_dim, 4, 2, 1, bias=False),  # Upsampling
        nn.BatchNorm2d(feature_dim),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(feature_dim, feature_dim, 4, 2, 1, bias=False),  # Upsampling
        nn.BatchNorm2d(feature_dim),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(feature_dim, num_channels, 4, 2, 1, bias=False),  # Final layer to generate image
        nn.Tanh()  # Output normalized to [-1, 1]
    )

  def forward(self, noise, text_embed):
    # Encode the text embedding
    encoded_text = self.text_encoder(text_embed)
    # Concatenate noise and encoded text, and reshape for input to the generator
    input = torch.cat([noise, encoded_text], dim=1).unsqueeze(2).unsqueeze(2)
    # Generate the image
    return self.main(input)

# num_channels=3, feature_dim=128, embedding_dim=768, reduced_dim_size=256
class Discriminator(nn.Module):
  def __init__(self, num_channels, feature_dim, embedding_dim, reduced_dim_size):
    super(Discriminator, self).__init__()
    self.reduced_dim_size = reduced_dim_size

    # Image encoder to extract features from the input image
    self.image_encoder = nn.Sequential(
        nn.Conv2d(num_channels, feature_dim, 4, 2, 1, bias=False),  # Downsampling
        nn.LeakyReLU(negative_slope=0.2, inplace=True),

        nn.Conv2d(feature_dim, feature_dim, 4, 2, 1, bias=False),  # Downsampling
        nn.LeakyReLU(negative_slope=0.2, inplace=True),

        nn.Conv2d(feature_dim, feature_dim*2, 4, 2, 1, bias=False),  # Downsampling
        nn.BatchNorm2d(feature_dim*2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),

        nn.Conv2d(feature_dim*2, feature_dim*4, 4, 2, 1, bias=False),  # Downsampling
        nn.BatchNorm2d(feature_dim*4),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),

        nn.Conv2d(feature_dim*4, feature_dim*8, 4, 2, 1, bias=False),  # Downsampling
        nn.BatchNorm2d(feature_dim*8),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
    )

    # Text encoder to reduce the dimensionality of the text embedding
    self.text_encoder = nn.Sequential(
        nn.Linear(in_features=embedding_dim, out_features=reduced_dim_size),
        nn.BatchNorm1d(num_features=reduced_dim_size),
        nn.LeakyReLU(negative_slope=0.2),
    )

    # Final block to classify the concatenated image and text features
    self.final_block = nn.Sequential(
        nn.Conv2d(feature_dim*8 + reduced_dim_size, 1, 4, 1, 0, bias=False),  # Output a single value
        nn.Sigmoid()  # Output normalized to [0, 1]
    )

  def forward(self, image, text_embed):
    # Encode the input image
    image_encoded = self.image_encoder(image)
    # Encode the text embedding
    text_encoded = self.text_encoder(text_embed)
    # Replicate and reshape the text encoding to match the image feature map dimensions
    replicated_text = text_encoded.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)
    # Concatenate the image and text features
    input = torch.cat([image_encoded, replicated_text], dim=1)
    # Classify the concatenated features
    output = self.final_block(input)
    return output.view(-1, 1), image_encoded