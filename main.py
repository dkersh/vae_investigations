import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Define VAE model
class VAE(nn.Module):
    def __init__(self, x_dim: int = 28 * 28, k: int = 400, z_dim: int = 20) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.k = k
        self.z_dim = z_dim

        # Define Encoder
        self.encoder = nn.Sequential(nn.Linear(self.x_dim, self.k), nn.ReLU())
        self.fc_mu = nn.Linear(self.k, self.z_dim)
        self.fc_logvar = nn.Linear(self.k, self.z_dim)

        # Define Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.k),
            nn.ReLU(),
            nn.Linear(self.k, self.x_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.encoder(x)
        mu = self.fc_mu(enc)
        logvar = self.fc_logvar(enc)

        return mu, logvar

    def reparametrise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparametrise(mu, logvar)
        x_recon = self.decode(z)

        return x_recon, mu, logvar


def vae_loss(
    recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


train_dataset = datasets.MNIST(
    root="./mnist_data/", train=True, transform=transforms.ToTensor(), download=True
)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataset = datasets.MNIST(
    root="./mnist_data/", train=False, transform=transforms.ToTensor(), download=True
)

# Define Model Object
vae = VAE()
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

epochs = 10
for e in range(epochs):
    vae.train()
    total_loss = 0

    for x, _ in train_dataloader:
        x = x.view(-1, 784)
        optimizer.zero_grad()
        x_recon, mu, logvar = vae(x)
        loss = vae_loss(x_recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {e + 1}, Loss: {total_loss / len(train_dataloader.dataset):.4f}")
