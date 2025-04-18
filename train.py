import time
import torch
import torch.nn as nn
from model.dcgan import Generator, Discriminator
from model.text_encoder import encode_caption
from dataset import CustomDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import *
from config import *    
from tqdm import tqdm

def train(generator, discriminator, dataloader, gen_optimizer, dis_optimizer, bce_loss, l2_loss, l1_loss):
    """
    Trains a GAN model consisting of a generator and a discriminator using the provided dataloader and optimizers.
    Args:
        generator (torch.nn.Module): The generator model responsible for creating fake images.
        discriminator (torch.nn.Module): The discriminator model responsible for distinguishing real and fake images.
        dataloader (torch.utils.data.DataLoader): DataLoader providing batches of real images, wrong images, and embedded captions.
        gen_optimizer (torch.optim.Optimizer): Optimizer for the generator.
        dis_optimizer (torch.optim.Optimizer): Optimizer for the discriminator.
        bce_loss (torch.nn.Module): Binary Cross-Entropy loss function for adversarial training.
        l2_loss (torch.nn.Module): L2 loss function for feature similarity between real and fake images.
        l1_loss (torch.nn.Module): L1 loss function for pixel-wise reconstruction between real and fake images.
    Training Process:
        - The discriminator is trained to distinguish between real, fake, and wrong images using BCE loss.
        - The generator is trained to:
            1. Fool the discriminator (adversarial loss).
            2. Generate images with similar features to real images (feature similarity loss).
            3. Reconstruct images close to real ones (pixel-wise reconstruction loss).
        - Losses are logged per epoch, and the model is saved after training.
    Returns:
        None: The function trains the models in-place and saves the generator to the specified path.
    Notes:
        - The generator loss combines adversarial loss, feature similarity loss (scaled by 100), and pixel-wise loss (scaled by 50).
        - The discriminator loss includes real image loss, fake image loss, and contrastive loss for wrong images.
        - Training progress is logged every 10 epochs or at the final epoch.
    """
    for epoch in tqdm(range(NUM_EPOCHS)):
        dis_losses, gen_losses = [], []
        start_time = time.time()

        for batch in dataloader:
            real_images = batch['image'].to(DEVICE)
            embed_caption = batch['embed_caption'].to(DEVICE)
            wrong_images = batch['wrong_image'].to(DEVICE)
            batch_size = real_images.size(0)

            real_labels = torch.full((batch_size,), REAL_LABEL, device=DEVICE)
            fake_labels = torch.full((batch_size,), FAKE_LABEL, device=DEVICE)

            # ----------------------------
            # Train Discriminator
            # ----------------------------
            dis_optimizer.zero_grad()

            # Generate fake images
            noise = torch.randn(batch_size, NOISE_DIM, device=DEVICE)
            fake_images = generator(noise, embed_caption)

            # Real image loss
            real_outputs, _ = discriminator(real_images, embed_caption)
            real_loss = bce_loss(real_outputs.squeeze(), real_labels)

            # Fake image loss
            fake_outputs, _ = discriminator(fake_images.detach(), embed_caption)
            fake_loss = bce_loss(fake_outputs.squeeze(), fake_labels)

            # Wrong image loss (contrastive)
            wrong_outputs, _ = discriminator(wrong_images, embed_caption)
            contrastive_loss = bce_loss(wrong_outputs.squeeze(), fake_labels)

            # Total discriminator loss
            dis_loss = real_loss + fake_loss + contrastive_loss
            dis_loss.backward()
            dis_optimizer.step()
            dis_losses.append(dis_loss.item())

            # ----------------------------
            # Train Generator
            # ----------------------------
            gen_optimizer.zero_grad()

            # Generate new fake images
            noise = torch.randn(batch_size, NOISE_DIM, device=DEVICE)
            fake_images = generator(noise, embed_caption)

            fake_outputs, fake_features = discriminator(fake_images, embed_caption)
            _, real_features = discriminator(real_images, embed_caption)

            # Feature similarity (l2), reconstruction (l1), adversarial loss
            act_fake = torch.mean(fake_features, dim=0)
            act_real = torch.mean(real_features, dim=0)

            adv_loss = bce_loss(fake_outputs.squeeze(), real_labels)
            feature_loss = l2_loss(act_fake, act_real)
            pixel_loss = l1_loss(fake_images, real_images)

            gen_loss = adv_loss + 100 * feature_loss + 50 * pixel_loss
            gen_loss.backward()
            gen_optimizer.step()
            gen_losses.append(gen_loss.item())

        # ----------------------------
        # Logging per epoch
        # ----------------------------
        elapsed_time = time.time() - start_time
        avg_dis_loss = sum(dis_losses) / len(dis_losses)
        avg_gen_loss = sum(gen_losses) / len(gen_losses)

        if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] | Time: {elapsed_time:.2f}s')
            print(f'  Discriminator Loss: {avg_dis_loss:.4f}')
            print(f'  Generator Loss:     {avg_gen_loss:.4f}')
    
    # Save model
    save_model(generator, SAVE_MODEL_PATH)

def main():
    torch.manual_seed(42)
    
    captions_folder = os.path.normpath(CAPTIONS_PATH)
    images_folder = os.path.normpath(IMAGES_PATH)
    captions = load_captions(captions_folder, images_folder)
    encoded_captions = encode_caption(captions)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], ([0.5]))
    ])

    dataset = CustomDataset(
        image_dir=images_folder,
        captions_dict=encoded_captions,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    generator = Generator(noise_dim=NOISE_DIM, feature_dim=128, num_channels=3,
                        embedding_dim=384, reduced_dim_size=256).to(DEVICE)
    discriminator = Discriminator(num_channels=3, feature_dim=128,
                                embedding_dim=384, reduced_dim_size=256).to(DEVICE)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    bce_loss = nn.BCELoss()
    l2_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()


    train(generator, discriminator, dataloader, gen_optimizer, dis_optimizer, bce_loss, l2_loss, l1_loss)

if __name__ == "__main__":
    main()