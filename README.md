# Text-to-Image Generation using BERT + DCGAN

## Introduction

This project implements a Text-to-Image generation pipeline that leverages both natural language processing and computer vision techniques. The model uses a pre-trained **BERT** for text encoding and a **DCGAN** for image synthesis. The training process is designed for educational purposes to help users understand how to combine language models with generative adversarial networks.

### What is BERT + DCGAN?

This model consists of two main components:

- **BERT (Bidirectional Encoder Representations from Transformers):**  
  Utilizes a pre-trained BERT model from HuggingFace Transformers to convert textual descriptions into rich embeddings.

- **DCGAN (Deep Convolutional GAN):**  
  Contains a **Generator (G)** and **Discriminator (D)**:
  - **Generator:** Generates realistic-looking images based on the noise vector and concatenated text embeddings.
  - **Discriminator:** Tries to distinguish between real images (from the dataset) and fake images produced by the generator, ensuring the generated images are coherent with the text inputs.

Both networks are trained in an adversarial manner, where the generator learns to produce convincing images while the discriminator becomes better at detecting fake images.

## Requirements

Ensure you have Python **3.12.4+** installed. Install the required dependencies by running:

```sh
pip install -r requirements.txt
```
## Project Structure
```bash
├── utils.py              # Utility functions
├── train.py              # Training script for DCGAN
├── prepare_data.py       # Data preprocessing and text encoding using BERT
├── generate.py           # Generate images from custom text inputs
├── dataset.py            # Custom PyTorch Dataset class for image-text pairs
├── config.py             # Configuration settings for model and training
└── model/
    ├── dcgan.py          # Definitions of DCGAN Generator and Discriminator
    └── text_encode.py    # BERT-based text encoder implementation
```
## Usage
1/ Prepare the data
```sh
python prepare_data.py
```
2/ Train the model
```sh
python train.py
```
3/ Generate images
```sh
python generate.py
```

## Notes

1/ This project is intended for educational and experimental purposes. The architecture and training procedure may require further refinements for production-level performance.

2/ Ensure that your dataset contains paired image-text data for the model to learn correct associations.

3/ Training GANs can be unstable — experimenting with various hyperparameters, loss functions, or regularization strategies is encouraged.

4/ Contributions, suggestions, and improvements are welcome.


 
