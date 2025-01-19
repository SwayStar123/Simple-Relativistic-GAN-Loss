import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

# Import the loss components from loss.py
from relativistic_loss.loss import (
    gan_loss,
    r1_penalty,
    r2_penalty,
    approximate_r1_loss,
    approximate_r2_loss
)

###############################################################################
# DCGAN-like Generator and Discriminator
###############################################################################
class DCGenerator(nn.Module):
    """
    Transposed Conv architecture for 1-channel 28x28 images (MNIST).
    """
    def __init__(self, z_dim=100, ngf=64, nc=1):
        super(DCGenerator, self).__init__()
        # Input: z of shape (batch_size, z_dim, 1, 1)
        # Output: (batch_size, nc=1, 28, 28)
        # We'll size it carefully for 28x28, though typically DCGAN is for 32x32.
        # We do a smaller version that ends up 28x28.
        self.net = nn.Sequential(
            # 1) 4x4
            nn.ConvTranspose2d(z_dim, ngf*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            # 2) 8x8
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            # 3) 14x14
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # 4) 28x28
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class DCDiscriminator(nn.Module):
    """
    Convolutional Discriminator for 1x28x28 MNIST images.
    Outputs a single scalar score per image.
    """
    def __init__(self, nc=1, ndf=64):
        super(DCDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),  # 14x14
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),  # 7x7
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 3, 2, 0, bias=False),  # 3x3
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, 1, 3, 1, 0, bias=False)  # 1x1
        )

    def forward(self, x):
        # net(x) -> shape (batch_size, 1, 1, 1)
        out = self.net(x)
        return out.view(-1)  # Flatten to (batch_size,)


###############################################################################
# Utility: train one setup
###############################################################################
def train_gan(
    use_approx=False,
    epochs=2,
    batch_size=64,
    lr=2e-4,
    z_dim=100,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # scale to [-1,1]
    ])
    dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 2) Create models
    gen = DCGenerator(z_dim=z_dim).to(device)
    disc = DCDiscriminator().to(device)

    # 3) Optimizers
    opt_g = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    # 4) Training loop
    print(f"---- Training with use_approx={use_approx} ----")
    for epoch in range(epochs):
        # tqdm progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)

            # -----------------
            # Train Discriminator
            # -----------------
            opt_d.zero_grad()

            # Basic GAN loss
            z = torch.randn(batch_size, z_dim, 1, 1, device=device)
            d_loss = -gan_loss(disc, gen, imgs, z)  # default logistic

            # Add penalty
            if use_approx:
                # approximate R1, R2
                r1 = approximate_r1_loss(disc, imgs, sigma=0.01)
                with torch.no_grad():
                    fake_imgs = gen(z)
                r2 = approximate_r2_loss(disc, fake_imgs, sigma=0.01)
            else:
                # exact R1, R2
                r1 = r1_penalty(disc, imgs, gamma=0.1)
                with torch.no_grad():
                    fake_imgs = gen(z)
                r2 = r2_penalty(disc, fake_imgs, gamma=0.1)

            d_total = d_loss + r1 + r2
            d_total.backward()
            opt_d.step()

            # -----------------
            # Train Generator
            # -----------------
            opt_g.zero_grad()
            z = torch.randn(batch_size, z_dim, 1, 1, device=device)
            g_loss = gan_loss(disc, gen, imgs, z)  # by default logistic
            g_loss.backward()
            opt_g.step()

            # Display short info in tqdm bar
            pbar.set_postfix({
                "d_loss": f"{d_loss.item():.3f}",
                "r1": f"{r1.item():.3f}",
                "r2": f"{r2.item():.3f}",
                "g_loss": f"{g_loss.item():.3f}"
            })

    return gen

###############################################################################
# Main: Train both setups and save 9x9 images from each
###############################################################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Train with real R1 + R2
    gen_real = train_gan(use_approx=False, epochs=10, device=device)

    # 2) Train with approximate R1 + R2
    gen_approx = train_gan(use_approx=True, epochs=10, device=device)

    # Generate a fixed batch of noise
    fixed_z = torch.randn(81, 100, 1, 1, device=device)  # 9x9 grid

    # Evaluate and save images for the real-penalty model
    gen_real.eval()
    with torch.no_grad():
        imgs_real = gen_real(fixed_z)
    grid = torchvision.utils.make_grid(imgs_real, nrow=9, normalize=True, scale_each=True)

    vutils.save_image(grid, "samples_real.png")
    print("Saved 9x9 grid to samples_real.png")

    # Evaluate and save images for the approximate-penalty model
    gen_approx.eval()
    with torch.no_grad():
        imgs_approx = gen_approx(fixed_z)
    grid = torchvision.utils.make_grid(imgs_approx, nrow=9, normalize=True, scale_each=True)
    vutils.save_image(grid, "samples_approx.png")
    print("Saved 9x9 grid to samples_approx.png")
