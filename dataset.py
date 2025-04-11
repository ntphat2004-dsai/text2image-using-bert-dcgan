from PIL import Image
from torch.utils.data import Dataset
import random
import os

class CustomDataset(Dataset):
    def __init__(self, image_dir, captions_dict, transform=None, max_length=1000, max_samples=None):
        self.img_dir = image_dir
        self.captions = captions_dict
        self.img_names = list(self.captions.keys())
        if max_samples:
            self.img_names = self.img_names[:max_samples]
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name + '.jpg')
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        encoded_caption = self.captions[img_name]["embed"]
        caption = self.captions[img_name]["text"]

        if isinstance(encoded_caption, (list, tuple)):
            encoded_caption = encoded_caption[:self.max_length]
        if isinstance(caption, str):
            caption = caption[:self.max_length]

        # Choose wrong image randomly
        wrong_index = index
        while wrong_index == index:
            wrong_index = random.randint(0, len(self.img_names) - 1)

        wrong_img_name = self.img_names[wrong_index]
        wrong_img_path = os.path.join(self.img_dir, wrong_img_name + '.jpg')
        wrong_image = Image.open(wrong_img_path).convert('RGB')
        if self.transform:
            wrong_image = self.transform(wrong_image)

        return {
            "image": image,
            "embed_caption": encoded_caption,
            "text": caption,
            "wrong_image": wrong_image
        }
