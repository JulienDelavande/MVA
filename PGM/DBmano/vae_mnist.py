from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import sys
from torchvision import datasets, transforms

# Import des modèles
from models.mnistGFZ import Generator as GFZ
from models.mnistDFZ import Generator as DFZ

dimZ = 64
dimH = 500
n_iter = 100
batch_size = 50
lr = 1e-4

def train(generator, data_loader, optimizer, loss_fn, device):
    generator.train()
    for epoch in range(n_iter):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Sample z from a normal distribution
            z = torch.randn(batch_size, dimZ).to(device)
            
            # Forward pass
            x_reconstructed = generator.pxz_params(z)
            
            # Compute loss
            loss = loss_fn(x_reconstructed, data)
            total_loss += loss.item()
            
            # Backward and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{n_iter}, Loss: {total_loss / len(data_loader)}")

def main(generator_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (1, 28, 28)
    dimY = 10
    n_channel = 64

    # Choix du générateur
    if generator_type == "GFZ":
        generator = GFZ(input_shape, dimH, dimZ, dimY, n_channel, 'sigmoid', 'GFZ').to(device)
    elif generator_type == "DFZ":
        generator = DFZ(input_shape, dimH, dimZ, dimY, n_channel, 'sigmoid', 'DFZ').to(device)
    else:
        raise ValueError("Invalid generator type. Choose 'GFZ' or 'DFZ'.")

    # Optimisateur et fonction de perte
    optimizer = optim.Adam(generator.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Chargement des données MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Entraînement
    train(generator, train_loader, optimizer, loss_fn, device)
    
    # Sauvegarde du modèle
    torch.save(generator.state_dict(), f"{generator_type}.pth")

if __name__ == "__main__":
    # Pass 'GFZ' or 'DFZ' as a command-line argument, e.g.:
    # python train.py GFZ
    # python train.py DFZ
    generator_type = sys.argv[1]  
    main(generator_type)
