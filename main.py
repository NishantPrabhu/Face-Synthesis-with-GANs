
# Main script to run DCGAN

import os 
import torch 
from tqdm import tqdm
from PIL import Image, ImageOps
from torchvision import transforms
from dcgan import DCGAN


# Transformation for image
# Already the right size so convert to Tensor and normalize
img_transform = transforms.Compose([
    transforms.ToTensor()
])


def prepare_input_data(img_dir, num_images=None):
    """ Aggregates images into a Tensor """

    filenames = os.listdir(img_dir)
    counter = 0
    images = []
    for name in tqdm(filenames):
        path = img_dir + '/' + name
        img = Image.open(path)
        img = ImageOps.grayscale(img)
        img = img_transform(img)
        img = 2.0 * img - 1.0
        images.append(img.numpy())
        counter += 1

        if num_images is not None and counter == num_images:
            break

    images = torch.FloatTensor(images)
    return images


# Main script

if __name__ == "__main__":

    img_dir = "../../images_small"

    images = prepare_input_data(img_dir=img_dir, num_images=None)

    # Training params
    epochs = 100
    steps_per_epoch = 500
    batch_size = 128
    save_freq = 5
    save_path = "../../saved_data"
    latent_dim = 200
    learning_rate = 5e-05
    n_critics = 5 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("")

    # Initialize GAN 
    gan = DCGAN(images, latent_dim, learning_rate, n_critics, device)

    # Train GAN
    gan.train(epochs, steps_per_epoch, batch_size, save_freq, save_path)
