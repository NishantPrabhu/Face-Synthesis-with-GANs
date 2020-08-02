
# DCGAN implementation in Torch

# Dependencies
import numpy as np
import torch
import time
import random
import pickle
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt


# Discriminator

class Discriminator(torch.nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_1 = torch.nn.Conv2d(1, 4, kernel_size=(3, 3), stride=(2, 2))
        self.dropout_1 = torch.nn.Dropout2d(0.2)
        self.conv_2 = torch.nn.Conv2d(4, 8, kernel_size=(3, 3), stride=(1, 1))
        self.dropout_2 = torch.nn.Dropout2d(0.2)
        self.conv_3 = torch.nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1))
        self.dropout_3 = torch.nn.Dropout2d(0.2)
        self.flatten = torch.nn.Flatten()
        self.fc_reduce = torch.nn.Linear(11664, 100)
        self.fc_map = torch.nn.Linear(100, 9)
        self.fc_out = torch.nn.Linear(9, 2)

    def forward(self, x):
        # In shape : (batch_size, 1, 64, 64)

        x = self.conv_1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout_1(x)
        x = self.conv_2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout_2(x)
        x = self.conv_3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout_3(x)
        x = self.flatten(x)
        x = self.fc_reduce(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_map(x)
        x = self.fc_out(x)
        x = F.log_softmax(x, dim=-1)

        return x

# Generator

class Generator(torch.nn.Module):

    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.fc_1 = torch.nn.Linear(latent_dim, 16*4*4)
        self.conv_trans_1 = torch.nn.ConvTranspose2d(
            16, 32, stride=(2, 2), kernel_size=(4, 4), padding=(1, 1)
        )
        self.conv_trans_2 = torch.nn.ConvTranspose2d(
            32, 64, stride=(2, 2), kernel_size=(4, 4), padding=(1, 1)
        )
        self.conv_trans_3 = torch.nn.ConvTranspose2d(
            64, 128, stride=(2, 2), kernel_size=(4, 4), padding=(1, 1)
        )
        self.conv_trans_4 = torch.nn.ConvTranspose2d(
            128, 128, stride=(2, 2), kernel_size=(4, 4), padding=(1, 1)
        )
        self.conv = torch.nn.Conv2d(
            128, 1, stride=(1, 1), kernel_size=(1, 1)
        )

    def forward(self, x):
        # Input shape : (batch_size, latent_dim)

        x = self.fc_1(x)
        x = F.leaky_relu(x, 0.2)
        x = torch.reshape(x, (-1, 16, 4, 4))
        x = self.conv_trans_1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv_trans_2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv_trans_3(x)
        x = F.leaky_relu(x)
        x = self.conv_trans_4(x)
        x = F.leaky_relu(x)
        x = self.conv(x)
        x = torch.tanh(x)

        return x


# GAN model

class GAN(torch.nn.Module):

    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()

        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        # Input shape : (batch_size, latent_dim)

        x = self.generator(x)
        x = self.discriminator(x)

        return x


# Main GANBuilder

class DCGAN():

    def __init__(self, data, latent_dim, learning_rate, n_critics, device):

        self.data = data
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.device = device
        self.n_critics = n_critics

        # Initialize generator, discriminator and GAN
        self.generator = Generator(latent_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.gan_model = GAN(self.generator, self.discriminator).to(self.device)

        # Create optimizers for discriminator and GAN
        self.gan_optim = optim.RMSprop(self.gan_model.parameters(), lr=self.lr)
        self.disc_optim = optim.RMSprop(self.discriminator.parameters(), lr=self.lr)

        # Pretrain discriminator
        self.pretrain_critics(data_size=1000, epochs=10)


    def generate_latent_samples(self, size):

        mat = np.random.normal(loc=0, scale=1, size=(size, self.latent_dim))
        return torch.FloatTensor(mat).to(self.device)


    def generate_disc_batch(self, size, pretraining=False):

        # Extract size//2 real samples from data
        idx = np.random.choice(
            np.arange(self.data.shape[0]), size=size//2, replace=True
        )
        true_data = torch.FloatTensor(self.data[idx]).to(self.device)

        # Generate size//2 fake samples using generator
        latent_ = self.generate_latent_samples(size=size//2)
        fake_data = self.generator(latent_)

        # Concat
        data = torch.cat((true_data, fake_data), dim=0)

        # Flip labels randomly to train discriminator better
        if random.random() < 0.9 or pretraining:
            true_labels = torch.LongTensor([0]*(size//2)).to(self.device)
            fake_labels = torch.LongTensor([1]*(size//2)).to(self.device)
        else:
            true_labels = torch.LongTensor([1]*(size//2)).to(self.device)
            fake_labels = torch.LongTensor([0]*(size//2)).to(self.device)

        return data, torch.cat((true_labels, fake_labels), dim=0)


    def generate_gan_batch(self, size):

        # Generate size latent samples and generate inverted labels
        data = self.generate_latent_samples(size)
        labels = torch.LongTensor([0]*size).to(self.device)

        return data, labels


    def train_model_on_batch(self, model, optimizer, x, y):

        optimizer.zero_grad()
        probs = model(x)
        loss = F.nll_loss(probs, y, reduction='mean')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()

        return loss.item()


    def pretrain_critics(self, data_size, epochs):

        print("\n[INFO] Pretraining critics...\n")

        for epoch in range(epochs):
            total_loss = 0

            for _ in range(self.n_critics):
                
                x, y = self.generate_disc_batch(data_size, True)
                loss = self.train_model_on_batch(
                    self.discriminator, self.disc_optim, x, y
                )
                total_loss += loss
               
            print("Epoch {} - Loss {:.4f}".format(
                epoch+1, total_loss/self.n_critics
            ))


    def discriminator_trainable(self, val):

        for param in self.discriminator.parameters():
            param.requires_grad = val


    def test_generator(self, num_samples):
        """ Accuracy of the discriminator on generator samples """

        # Generate num_samples//2 real images and fake images
        latent_ = self.generate_latent_samples(num_samples)
        labels = torch.LongTensor([1]*num_samples).to(self.device)
        probs = self.gan_model(latent_)
        preds = probs.argmax(dim=-1)
        correct = preds.eq(labels).sum().item()
        print("\n[TESTING] GAN accuracy: {:.4f}".format(correct/num_samples))

        # Show some images
        gen_out = self.generator(latent_)
        images = gen_out.cpu().detach().numpy()[:5] 
        images = (images + 1.) / 2.0
        
        fig = plt.figure(figsize=(5, 2))
        for i in range(5):
            fig.add_subplot(1, 5, i+1)
            plt.axis("off")
            plt.imshow(images[i].squeeze(0), cmap='gray')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(3)
        plt.close()


    def train(self, epochs, steps_per_epoch, batch_size, save_freq, save_path):

        disc_losses = [] 
        gan_losses = []

        for epoch in range(epochs):
            print("\n\nEpoch {}/{}".format(epoch+1, epochs))
            print("--------------------------------------")

            for step in range(steps_per_epoch):

                disc_loss_hist = []

                for _ in range(self.n_critics):
                    
                    # Generate batch for discriminator
                    x, y = self.generate_disc_batch(batch_size)

                    # Train discriminator on fake batch
                    disc_loss = self.train_model_on_batch(
                        self.discriminator, self.disc_optim, x, y
                    )
                    disc_loss_hist.append(disc_loss)

                # Render discriminator untrainable
                self.discriminator_trainable(False)

                # Generate batch for GAN
                x_gan, y_gan = self.generate_gan_batch(batch_size)

                # Train GAN on this batch
                gan_loss = self.train_model_on_batch(
                    self.gan_model, self.gan_optim, x_gan, y_gan
                )

                # Render discriminator trainable
                self.discriminator_trainable(True)

                if step % 100 == 0:
                    print("Step {:3d} - Disc loss {:.4f} - GAN loss {:.4f}".format(
                        step, sum(disc_loss_hist)/self.n_critics, gan_loss
                    ))

                # Update loss logs
                disc_losses.append(sum(disc_loss_hist)/self.n_critics)
                gan_losses.append(gan_loss)

            # Save model
            if epoch % save_freq == 0:
                torch.save(self.generator.state_dict(), save_path +
                           "/generator_{}".format(epoch+1))
                torch.save(self.gan_model.state_dict(),
                           save_path + "/gan_{}".format(epoch+1))

            # Test generator
            self.test_generator(num_samples=50)

            print("====================================")

        # Save critic and GAN losses
        with open(save_path + "/disc_loss.pkl", "wb") as f:
            pickle.dump(disc_losses, f)

        with open(save_path + "/gan_loss.pkl", "wb") as f:
            pickle.dump(gan_losses, f)

