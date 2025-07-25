# imgcapgen/models/scratchCNN_LSTM.py
"""
Simple CNN (from scratch) + LSTM image captioning model.

CNN Encoder built with Conv2d layers, no pretrained backbone.
RNN Decoder built with Embedding + LSTM + FCN for vocabulary prediction.
"""

import torch
import torch.nn as nn

# CNN encoder (from scratch) and RNN decoder (LSTM)
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),  # output: [B, 512, 1, 1]
        )
        self.embed = nn.Linear(512, embed_size)

    def forward(self, images):
        features = self.cnn(images)               # [B, 512, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 512]
        features = self.embed(features)           # [B, embed_size]
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fcn = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, features, captions):
        embeds = self.embedding(captions[:, :-1])
        x = torch.cat((features.unsqueeze(1), embeds), dim=1)
        x, _ = self.lstm(x)
        x = self.fcn(self.drop(x))
        return x

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)

    def generate_caption(self, inputs, hidden=None, max_len=20, vocab=None):
        batch_size = inputs.size(0)
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
        if hidden is None:
            hidden = self.init_hidden(batch_size, inputs.device)
    
        captions = []
        for _ in range(max_len):
            output, hidden = self.lstm(inputs, hidden)
            output = self.fcn(output.squeeze(1))
            predicted = output.argmax(dim=1)
            captions.append(predicted.item())
            
            if vocab and vocab.itos[predicted.item()] == "<EOS>":
                break
            inputs = self.embedding(predicted)
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(1)
        return [vocab.itos[idx] for idx in captions] if vocab else captions


class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3):
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, drop_prob)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
