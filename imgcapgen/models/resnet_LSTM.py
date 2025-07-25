# imgcapgen/models/resnet_LSTM.py
"""
ResNet-50 (pretrained) + LSTM image captioning model.

CNN Encoder: frozen ResNet-50 backbone, linear projection to embed size.
RNN Decoder: Embedding + LSTM + FCN to predict vocab indices.
"""

import torch
import torch.nn as nn
import torchvision.models as models

#CNN encoder (ResNet-50) and RNN decoder (LSTM) architecture
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]  # remove final FC layer
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)                  # [B, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 2048]
        features = self.embed(features)                 # [B, embed_size]
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
        embeds = self.embedding(captions[:, :-1])               # [B, seq_len-1, embed_size]
        x = torch.cat((features.unsqueeze(1), embeds), dim=1)   # [B, seq_len, embed_size]
        x, _ = self.lstm(x)
        x = self.fcn(self.drop(x))                             # [B, seq_len, vocab_size]
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
