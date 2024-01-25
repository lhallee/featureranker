import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class Autoencoder(nn.Module):
    def __init__(self, config):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(config.input_size, config.hidden_size)
        self.label_bottleneck = nn.Linear(config.hidden_size, config.L)
        self.l1_bottleneck = nn.Linear(config.hidden_size, config.hidden_size - config.L)
        self.decoder = nn.Linear(config.hidden_size, config.input_size)
        self.gelu = nn.GELU()

    def forward(self, input):
        hidden = self.encoder(input)
        hidden = self.gelu(hidden) # batch_size, hidden_size

        bottleneck = self.label_bottleneck(hidden) # batch_size, L
        bottleneck = self.gelu(bottleneck)
        
        l1 = self.l1_bottleneck(hidden) # batch_size, hidden_size - L
        l1_gelu = self.gelu(l1)

        feature = torch.cat([bottleneck, l1_gelu], dim=-1) # batch_size, hidden_size
        output = self.decoder(feature)
        return output, feature, l1


def train_model(self, dataloader, optimizer, num_epochs, alpha=1.0):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, input in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')):
            output, feature, l1 = self.forward(input)
            l1_penalty = alpha * torch.norm(l1, 1) # if l1 == 0 everywhere, l1_penalty is 0
            loss = self.criterion(output, input) + l1_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i+1) % 100 == 0:
                avg_loss = running_loss / 100
                tqdm.set_postfix_str(f'Average Loss: {avg_loss:.4f}')
                running_loss = 0.0

        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')