# imgcapgen/models/resnet_finetune_LSTM.py
"""
ResNet-50 (partially fine-tuned on layer3/layer4) + LSTM image captioning model.

Encoder: ResNet-50 pretrained, unfreezes last two blocks, with projection + BN + Dropout.
Decoder: Embedding + LSTM + FCN.
"""

import torch
import torch.nn as nn
import torchvision.models as models

# CNN encoder (ResNet-50) and RNN decoder (LSTM)
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, drop_prob=0.3):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)

        # Freeze all layers
        for param in resnet.parameters():
            param.requires_grad = False

        # Unfreeze layer3 & layer4
        for name, param in resnet.named_parameters():
            if "layer3" in name or "layer4" in name:
                param.requires_grad = True

        modules = list(resnet.children())[:-1]  # remove final FC
        self.resnet = nn.Sequential(*modules)

        # Add projection head with dropout & batchnorm
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, images):
        features = self.resnet(images)                 # [B, 2048, 1, 1]
        features = features.view(features.size(0), -1) # [B, 2048]
        features = self.embed(features)                # [B, embed_size]
        features = self.bn(features)
        features = self.drop(features)
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
        # make sure hidden state is always correct shape
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
        self.encoder = EncoderCNN(embed_size, drop_prob)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, drop_prob)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
