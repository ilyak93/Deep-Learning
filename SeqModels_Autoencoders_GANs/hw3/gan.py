from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        modules = []

        modules.append(nn.Conv2d(in_size[0], 128, kernel_size=5, stride=2, padding=2))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.LeakyReLU())

        modules.append(nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2))
        modules.append(nn.BatchNorm2d(256))
        modules.append(nn.LeakyReLU())

        modules.append(nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2))
        modules.append(nn.BatchNorm2d(512))
        modules.append(nn.LeakyReLU())

        modules.append(nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=2))
        modules.append(nn.BatchNorm2d(1024))
        modules.append(nn.LeakyReLU())

        self.cnn = nn.Sequential(*modules)

        # Fully connected part
        modules = []

        modules.append(nn.Linear(int((self.in_size[1] / 16) * (self.in_size[2] / 16)) * 1024, 128))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Linear(128, 1))

        self.fc = nn.Sequential(*modules)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        y = self.cnn(x)
        y = y.view(-1, y.size(1)*y.size(2)*y.size(3))
        y = self.fc(y)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        self.featuremap_size = featuremap_size

        # First layer - fully connected
        self.fc = nn.Linear(z_dim, featuremap_size * featuremap_size * 1024)                

        # CNN part
        modules = []

        modules.append(nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1))
        modules.append(nn.BatchNorm2d(512))
        modules.append(nn.LeakyReLU())

        modules.append(nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1))
        modules.append(nn.BatchNorm2d(256))
        modules.append(nn.LeakyReLU())

        modules.append(nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.LeakyReLU())

        modules.append(nn.ConvTranspose2d(128, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1))
        modules.append(nn.Tanh())

        self.cnn = nn.Sequential(*modules)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        z = self.fc(z)
        z = z.view(-1, 1024, self.featuremap_size, self.featuremap_size)
        x = self.cnn(z)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    
    # Create targets (labels) of generated images and dataset images, with noise
    noisy_y_data = data_label + (torch.rand_like(y_data, device = y_data.device) * label_noise - label_noise / 2)
    noisy_y_generated = (1-data_label)+(torch.rand_like(y_generated, device = y_generated.device) * label_noise - label_noise / 2)
    # BCE loss with Sigmoid
    loss_fn = nn.BCEWithLogitsLoss()
    # Compute loss with respect to generated data and with respect to dataset data
    loss_data = loss_fn(y_data, noisy_y_data)
    loss_generated = loss_fn(y_generated, noisy_y_generated)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    new_data_label = torch.ones_like(y_generated, device=y_generated.device) * data_label
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(y_generated, new_data_label)
    # ========================
    return loss


def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_optimizer.zero_grad()

    #  1. Show the discriminator real and generated data
    dsc_real_res = dsc_model.forward(x_data)
    FakeData = gen_model.sample(x_data.shape[0], with_grad=False)
    dsc_fake_res = dsc_model.forward(FakeData)

    #  2. Calculate discriminator loss
    dsc_loss = dsc_loss_fn(dsc_real_res, dsc_fake_res)

    #  3. Update discriminator parameters
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()

    #  1. Show the discriminator generated data
    FakeData = gen_model.sample(x_data.shape[0], with_grad=True)
    dsc_fake_res = dsc_model.forward(FakeData)

    #  2. Calculate generator loss
    gen_loss = gen_loss_fn(dsc_fake_res)

    #  3. Update generator parameters
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f'{checkpoint_file}.pt'

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    avg_gen_dsc_losses = [(dsc_loss+gen_loss)/2 for dsc_loss, gen_loss in zip(dsc_losses, gen_losses)]

    if ((gen_losses[-1]+dsc_losses[-1])/2 == min(avg_gen_dsc_losses) or len(gen_losses) == 1):
        saved = True

    # Save model checkpoint if requested
    if saved == True and checkpoint_file is not None:
        torch.save(gen_model, checkpoint_file)
    # ========================

    return saved
