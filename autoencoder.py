import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    """
    Autoencoder neural network model for feature learning.

    Parameters:
    - input_dim (int): Dimensionality of the input features.
    - latent_dim (int): Dimensionality of the latent space.
    """
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
            nn.ReLU(inplace=True),
        )
        # Decoder architecture
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    

def trainAutoencoder(model, train_loader, val_loader, num_epochs, name):
 
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-8)
    train_loss = []
    val_final_loss = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0.0

        for batch_idx, data in enumerate(train_loader):
            data = data[0]
            output = model(data)
            loss = loss_fn(output, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss

        avg_train_loss = total_train_loss

        train_loss.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for val_batch_idx, (val_data) in enumerate(val_loader):
                val_data = val_data[0]
                val_output = model(val_data)

                val_loss = loss_fn(val_output, val_data)
                total_val_loss += val_loss

        avg_val_loss = total_val_loss
        val_final_loss.append(avg_val_loss)

        print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(
            epoch + 1, num_epochs, avg_train_loss, avg_val_loss))

        before_lr = optimizer.param_groups[0]["lr"]
        after_lr = optimizer.param_groups[0]["lr"]
        if before_lr != after_lr:
            print("Epoch %d: Adam lr %.8f -> %.8f" % (epoch, before_lr, after_lr))

    torch.save(model.state_dict(), "autoencoder" + name + '.pth')
    print(model.encoder[0].weight.detach().numpy())
